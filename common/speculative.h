#pragma once

#include "llama.h"
#include "common.h"

#include <cmath>
#include <random>

struct common_speculative;

enum common_speculative_proposal_mode {
    COMMON_SPECULATIVE_PROPOSAL_STOCHASTIC,
    COMMON_SPECULATIVE_PROPOSAL_GREEDY,
};

struct common_speculative_token_dist {
    llama_tokens ids;
    std::vector<float> probs;
    // Keep mode after the legacy aggregate fields so existing { ids, probs } initializers
    // remain source-compatible and default to stochastic verification.
    common_speculative_proposal_mode mode = COMMON_SPECULATIVE_PROPOSAL_STOCHASTIC;

    bool valid() const {
        switch (mode) {
            case COMMON_SPECULATIVE_PROPOSAL_STOCHASTIC: {
                if (ids.empty() || ids.size() != probs.size()) {
                    return false;
                }
                float sum = 0.0f;
                for (float p : probs) {
                    if (!std::isfinite(p) || p < 0.0f) {
                        return false;
                    }
                    sum += p;
                }
                return std::isfinite(sum) && sum > 0.0f;
            }
            case COMMON_SPECULATIVE_PROPOSAL_GREEDY:
                return ids.empty() && probs.empty();
        }
        return false;
    }
};

void common_speculative_dist_condition_on_p_min(common_speculative_token_dist & dist, float p_min);

bool common_speculative_needs_checkpoint(common_context_seq_rm_type seq_rm_type, uint32_t n_rs_seq, size_t n_mutated);

bool common_speculative_can_verify_stochastic(common_context_seq_rm_type seq_rm_type,
                                              uint32_t                   n_rs_seq,
                                              size_t                     n_draft,
                                              size_t                     n_dists,
                                              float                      temperature,
                                              float                      p_min,
                                              int32_t                    n_min);

float common_speculative_dflash_normalized_entropy(const std::vector<float> & probs);
bool common_speculative_dflash_use_greedy(
        const std::vector<float> & probs,
        float greedy_max_entropy);
void common_speculative_dflash_select(
        const float * lattice,
        int32_t n_block_tokens,
        int32_t n_embd,
        int32_t selector_top_k,
        float temperature,
        float greedy_max_entropy,
        float p_min,
        int32_t n_min,
        std::mt19937 & rng,
        llama_tokens & result,
        std::vector<common_speculative_token_dist> & dists);
void common_speculative_truncate_draft(
        llama_tokens & result,
        std::vector<common_speculative_token_dist> * dists,
        int32_t n_max);

// comma separated list the provided types
std::string common_speculative_type_name_str(const std::vector<enum common_speculative_type> & types);

// comma separated list of all types
const char * common_speculative_all_types_str();

// parse user provided types
std::vector<enum common_speculative_type> common_speculative_types_from_names(const std::vector<std::string> & names);

// infer the spec types from the GGUF metadata of a draft model; empty if unknown
std::vector<enum common_speculative_type> common_speculative_types_from_gguf(const std::string & path);

// convert string to type
enum common_speculative_type common_speculative_type_from_name(const std::string & name);

// convert type to string
std::string common_speculative_type_to_str(enum common_speculative_type type);

// return the max number of draft tokens based on the speculative parameters
int32_t common_speculative_n_max(const common_params_speculative * spec);

// return the maximum number of draft-context positions that draft-model processing can mutate
size_t common_speculative_draft_max_mutation_span(const common_params_speculative * spec);

common_params common_base_params_to_speculative(const common_params & params);

struct common_speculative_output_limits {
    int32_t total;
    int32_t per_seq;
};

// return the output limits needed for speculative decoding
common_speculative_output_limits common_speculative_get_output_limits(
        int32_t n_batch, int32_t n_parallel, int32_t n_draft);

common_speculative * common_speculative_init(common_params_speculative & params, uint32_t n_seq);

void common_speculative_free(common_speculative * spec);

struct common_speculative_draft_params {
    // this flag is used to chain the drafts through all the available implementations
    // after the first successful draft from an implementation, we set it
    //   to false to prevent further drafts for that sequence
    // at the end of the draft() call, all drafting flags will be reset to false
    bool drafting = false;

    // overrides individual configurations (-1 disabled)
    // can be used to constraint the max draft based on the remaining context size
    int32_t n_max = -1;

    llama_pos   n_past;
    llama_token id_last;

    // TODO: remove in the future by keeping track of the prompt from the _begin() call and the consecutive accept calls
    const llama_tokens * prompt;

    // the generated draft from the last _draft() call
    llama_tokens * result;

    // optional sparse proposal distributions, one per draft token
    std::vector<common_speculative_token_dist> * dists = nullptr;

    float temperature = 0.0f;
    uint32_t seed = LLAMA_DEFAULT_SEED;
};

common_speculative_draft_params & common_speculative_get_draft_params(common_speculative * spec, llama_seq_id seq_id);

// optionally call once at the beginning of a new generation
void common_speculative_begin(common_speculative * spec, llama_seq_id seq_id, const llama_tokens & prompt);

// process the batch and update the internal state of the speculative context
bool common_speculative_process(common_speculative * spec, const llama_batch & batch);

// generate drafts for the sequences specified with `common_speculative_get_draft_params`
void common_speculative_draft(common_speculative * spec);

// informs the speculative context that n_accepted tokens were accepted by the target model
void common_speculative_accept(common_speculative * spec, llama_seq_id, uint16_t n_accepted);

// (optional) get/set internal state
bool common_speculative_get_state(common_speculative * spec, llama_seq_id seq_id, std::vector<uint8_t> & data);
void common_speculative_set_state(common_speculative * spec, llama_seq_id seq_id, const std::vector<uint8_t> & data);

// print statistics about the speculative decoding
void common_speculative_print_stats(const common_speculative * spec);

struct common_speculative_deleter {
    void operator()(common_speculative * s) { common_speculative_free(s); }
};

typedef std::unique_ptr<common_speculative, common_speculative_deleter> common_speculative_ptr;

struct common_speculative_init_result {
    common_speculative_init_result(common_params & params, llama_model * model_tgt, llama_context * ctx_tgt);
    ~common_speculative_init_result();

    llama_model   * model();
    llama_context * context();

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

using common_speculative_init_result_ptr = std::unique_ptr<common_speculative_init_result>;

common_speculative_init_result_ptr common_speculative_init_from_params(common_params & params, llama_model * model_tgt, llama_context * ctx_tgt);
