#include "speculative.h"
#include "sampling.h"

#undef NDEBUG

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <random>
#include <utility>
#include <vector>

static void select_stochastic_reference(
        const float * lattice,
        int32_t n_block_tokens,
        int32_t n_embd,
        int32_t selector_top_k,
        float temperature,
        float p_min,
        std::mt19937 & rng,
        llama_tokens & result,
        std::vector<common_speculative_token_dist> & dists) {
    int32_t predecessor = 0;
    for (int32_t i = 1; i < n_block_tokens; ++i) {
        const float * row = lattice + (size_t) i * n_embd;
        const float * scores = row + selector_top_k + (size_t) predecessor * selector_top_k;

        common_speculative_token_dist dist;
        dist.ids.resize(selector_top_k);
        dist.probs.resize(selector_top_k);
        const float max_score = *std::max_element(scores, scores + selector_top_k);
        float sum = 0.0f;
        for (int32_t k = 0; k < selector_top_k; ++k) {
            dist.ids[k] = (llama_token) row[k];
            dist.probs[k] = std::exp((scores[k] - max_score) / temperature);
            sum += dist.probs[k];
        }
        for (float & p : dist.probs) {
            p /= sum;
        }
        std::discrete_distribution<int32_t> sample(dist.probs.begin(), dist.probs.end());
        predecessor = sample(rng);
        if (dist.probs[predecessor] < p_min) {
            break;
        }
        result.push_back(dist.ids[predecessor]);
        dists.push_back(std::move(dist));
    }
}

static void test_entropy_policy() {
    const std::vector<float> sharp   = { 1.0f, 0.0f, 0.0f };
    const std::vector<float> uniform = { 1.0f / 3.0f, 1.0f / 3.0f, 1.0f / 3.0f };

    assert(std::fabs(common_speculative_dflash_normalized_entropy(sharp)) < 1e-6f);
    assert(std::fabs(common_speculative_dflash_normalized_entropy(uniform) - 1.0f) < 1e-6f);
    assert(!common_speculative_dflash_use_greedy(sharp, -1.0f));
    assert(common_speculative_dflash_use_greedy(sharp, 0.0f));
    assert(!common_speculative_dflash_use_greedy(uniform, 0.99f));
}

static void test_mixed_greedy_and_stochastic_rows_keep_distributions_aligned() {
    constexpr int32_t n_block_tokens = 3;
    constexpr int32_t n_embd         = 6;
    constexpr int32_t top_k          = 2;
    std::vector<float> lattice(n_block_tokens * n_embd, 0.0f);

    float * sharp = lattice.data() + n_embd;
    sharp[0] = 10.0f;
    sharp[1] = 11.0f;
    sharp[2] = 10.0f;
    sharp[3] =  0.0f;

    float * flat = lattice.data() + 2 * n_embd;
    flat[0] = 20.0f;
    flat[1] = 21.0f;
    flat[2] =  0.0f;
    flat[3] =  0.0f;

    std::mt19937 rng(123);
    llama_tokens result;
    std::vector<common_speculative_token_dist> dists;
    common_speculative_dflash_select(
            lattice.data(), n_block_tokens, n_embd, top_k,
            1.0f, 0.1f, 0.0f, 0, rng, result, dists);

    assert(result.size() == 2);
    assert(result[0] == 10);
    assert(dists.size() == result.size());
    assert(dists[0].mode == COMMON_SPECULATIVE_PROPOSAL_GREEDY);
    assert(dists[0].ids.empty());
    assert(dists[0].probs.empty());
    assert(dists[1].mode == COMMON_SPECULATIVE_PROPOSAL_STOCHASTIC);
    assert((dists[1].ids == llama_tokens { 20, 21 }));
    assert(std::fabs(dists[1].probs[0] - 0.5f) < 1e-6f);
    assert(std::fabs(dists[1].probs[1] - 0.5f) < 1e-6f);
    assert(result[1] == 20 || result[1] == 21);
}

static llama_token_data_array make_target_dist(std::vector<llama_token_data> & data) {
    return { data.data(), data.size(), -1, true };
}

static void test_mixed_verifier_modes() {
    common_speculative_token_dist greedy;
    greedy.mode = COMMON_SPECULATIVE_PROPOSAL_GREEDY;

    common_speculative_token_dist stochastic;
    stochastic.ids   = { 20, 21 };
    stochastic.probs = { 0.5f, 0.5f };

    std::vector<llama_token_data> target_data = {
        { 20, 0.0f, 0.5f },
        { 21, 0.0f, 0.5f },
    };
    auto target = make_target_dist(target_data);
    std::mt19937 rng(123);

    const auto greedy_accept = common_speculative_verify_token(10, greedy, 10, nullptr, rng);
    assert(greedy_accept.accepted);
    assert(greedy_accept.token == 10);

    const auto stochastic_after_greedy =
            common_speculative_verify_token(20, stochastic, 20, &target, rng);
    assert(stochastic_after_greedy.accepted);
    assert(stochastic_after_greedy.token == 20);

    const auto stochastic_first =
            common_speculative_verify_token(21, stochastic, 21, &target, rng);
    assert(stochastic_first.accepted);
    assert(stochastic_first.token == 21);

    const auto greedy_reject = common_speculative_verify_token(10, greedy, 11, nullptr, rng);
    assert(!greedy_reject.accepted);
    assert(greedy_reject.token == 11);
}

static void test_malformed_proposal_modes_are_invalid() {
    common_speculative_token_dist empty_stochastic;
    assert(!empty_stochastic.valid());

    common_speculative_token_dist populated_greedy;
    populated_greedy.mode  = COMMON_SPECULATIVE_PROPOSAL_GREEDY;
    populated_greedy.ids   = { 1 };
    populated_greedy.probs = { 1.0f };
    assert(!populated_greedy.valid());

    common_speculative_token_dist mismatched;
    mismatched.ids = { 1, 2 };
    mismatched.probs = { 1.0f };
    assert(!mismatched.valid());

    common_speculative_token_dist unknown_mode;
    unknown_mode.mode  = static_cast<common_speculative_proposal_mode>(-1);
    unknown_mode.ids   = { 1 };
    unknown_mode.probs = { 1.0f };
    assert(!unknown_mode.valid());

    for (const std::vector<float> & probs : {
            std::vector<float>{ -0.1f, 1.1f },
            std::vector<float>{ 0.0f, 0.0f },
            std::vector<float>{ std::numeric_limits<float>::quiet_NaN(), 1.0f },
            std::vector<float>{ std::numeric_limits<float>::infinity(), 1.0f } }) {
        common_speculative_token_dist invalid_probs;
        invalid_probs.ids   = { 1, 2 };
        invalid_probs.probs = probs;
        assert(!invalid_probs.valid());
    }
}

static void test_draft_limit_keeps_distributions_aligned() {
    llama_tokens result = { 10, 11, 12 };
    std::vector<common_speculative_token_dist> dists(3);
    for (size_t i = 0; i < dists.size(); ++i) {
        dists[i].ids   = { result[i] };
        dists[i].probs = { 1.0f };
    }

    common_speculative_truncate_draft(result, &dists, 1);
    assert((result == llama_tokens { 10 }));
    assert(dists.size() == result.size());
    assert(dists[0].mode == COMMON_SPECULATIVE_PROPOSAL_STOCHASTIC);
    assert((dists[0].ids == llama_tokens { 10 }));
}

static void test_early_stop_and_minimum_clear_both_outputs() {
    constexpr int32_t n_block_tokens = 2;
    constexpr int32_t n_embd         = 6;
    constexpr int32_t top_k          = 2;
    std::vector<float> lattice(n_block_tokens * n_embd, 0.0f);
    float * row = lattice.data() + n_embd;
    row[0] = 10.0f;
    row[1] = 11.0f;

    for (const auto [p_min, n_min] : { std::pair<float, int32_t>{ 0.6f, 0 }, { 0.0f, 2 } }) {
        std::mt19937 rng(123);
        llama_tokens result;
        std::vector<common_speculative_token_dist> dists;
        common_speculative_dflash_select(
                lattice.data(), n_block_tokens, n_embd, top_k,
                1.0f, -1.0f, p_min, n_min, rng, result, dists);
        assert(result.empty());
        assert(dists.empty());
    }
}

static void test_disabled_policy_matches_stochastic_reference_and_rng_state() {
    constexpr int32_t n_block_tokens = 5;
    constexpr int32_t top_k          = 3;
    constexpr int32_t n_embd         = top_k * (top_k + 1);
    std::mt19937 fixture_rng(0x5eed);
    std::uniform_real_distribution<float> score(-2.0f, 2.0f);

    for (int32_t trial = 0; trial < 64; ++trial) {
        std::vector<float> lattice(n_block_tokens * n_embd, 0.0f);
        for (int32_t i = 1; i < n_block_tokens; ++i) {
            float * row = lattice.data() + (size_t) i * n_embd;
            for (int32_t k = 0; k < top_k; ++k) {
                row[k] = (float) (trial * 100 + i * 10 + k);
            }
            for (int32_t k = top_k; k < n_embd; ++k) {
                row[k] = score(fixture_rng);
            }
        }

        const float temperature = 0.35f + 0.30f * (trial % 4);
        const float p_min = (trial % 3 == 0) ? 0.0f : (trial % 3 == 1 ? 0.2f : 0.4f);
        std::mt19937 expected_rng(7000 + trial);
        std::mt19937 actual_rng  (7000 + trial);
        llama_tokens expected;
        llama_tokens actual;
        std::vector<common_speculative_token_dist> expected_dists;
        std::vector<common_speculative_token_dist> actual_dists;

        select_stochastic_reference(
                lattice.data(), n_block_tokens, n_embd, top_k, temperature, p_min,
                expected_rng, expected, expected_dists);
        common_speculative_dflash_select(
                lattice.data(), n_block_tokens, n_embd, top_k, temperature, -1.0f, p_min, 0,
                actual_rng, actual, actual_dists);

        assert(actual == expected);
        assert(actual_dists.size() == expected_dists.size());
        for (size_t i = 0; i < actual_dists.size(); ++i) {
            assert(actual_dists[i].mode == COMMON_SPECULATIVE_PROPOSAL_STOCHASTIC);
            assert(actual_dists[i].ids == expected_dists[i].ids);
            assert(actual_dists[i].probs == expected_dists[i].probs);
        }
        assert(actual_rng == expected_rng);
    }
}

int main() {
    test_entropy_policy();
    test_mixed_greedy_and_stochastic_rows_keep_distributions_aligned();
    test_mixed_verifier_modes();
    test_malformed_proposal_modes_are_invalid();
    test_draft_limit_keeps_distributions_aligned();
    test_early_stop_and_minimum_clear_both_outputs();
    test_disabled_policy_matches_stochastic_reference_and_rng_state();
}
