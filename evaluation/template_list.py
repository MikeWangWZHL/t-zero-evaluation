template_list = {
    ("super_glue", "rte"): [
        "MNLI crowdsource",
        "guaranteed true",
        "can we infer",
        "GPT-3 style",
        "does this imply",
        "should assume",
        "does it follow that",
        "based on the previous passage",
        "justified in saying",
        "must be true",
    ],
    ("super_glue", "cb"): [
        "can we infer",
        "based on the previous passage",
        "claim true/false/inconclusive",
        "does it follow that",
        "justified in saying",
        "always/sometimes/never",
        "GPT-3 style",
        "consider always/sometimes/never",
        "guaranteed true",
        "must be true",
        "guaranteed/possible/impossible",
        "does this imply",
        "MNLI crowdsource",
        "should assume",
        "take the following as truth",
    ],
    ("anli", None): [
        "MNLI crowdsource",
        "should assume",
        "does it follow that",
        "GPT-3 style",
        "based on the previous passage",
        "justified in saying",
        "take the following as truth",
        "must be true",
        "can we infer",
        "guaranteed/possible/impossible",
        "always/sometimes/never",
        "does this imply",
        "consider always/sometimes/never",
        "claim true/false/inconclusive",
        "guaranteed true",
    ],
    ("super_glue", "wsc.fixed"): [
        "does the pronoun refer to",
        "by p they mean",
        "in other words",
        "I think they mean",
        "does p stand for",
        "GPT-3 Style",
        "replaced with",
        "p is/are r",
        "the pronoun refers to",
        "Who or what is/are",
    ],
    ("winogrande", "winogrande_xl"): [
        "does underscore refer to",
        "stand for",
        "underscore refer to",
        "fill in the blank",
        "Replace",
    ],
    ("story_cloze", "2016"): [
        "Answer Given options",
        "Choose Story Ending",
        "Movie What Happens Next",
        "Story Continuation and Options",
        "Novel Correct Ending",
    ],
    ("super_glue", "wic"): [
        "question-context-meaning-with-label",
        "grammar_homework",
        "affirmation_true_or_false",
        "same_sense",
        "GPT-3-prompt-with-label",
        "polysemous",
    ],
    ("hellaswag", None): [
        "Predict ending with hint",
        "Randomized prompts template",
        "complete_first_then",
        "if_begins_how_continues",
    ],
    ("super_glue", "copa"): [
        "exercise",
        "i_am_hesitating",
        "plausible_alternatives",
        "C1 or C2? premise, so/because???",
        "best_option",
        "more likely",
        "cause_effect",
        "choose",
    ],
    ("openbookqa","main"):[
        'choose_an_answer_with_options',
        'which_correct',
        'pick_using_id',
        'choices',
        'only_options',
        'which_correct_inverse',
        'pick_answer_with_options'
    ],
    ("piqa",None):[
        'choose the most appropriate solution',
        'finish_sentence_with_correct_choice',
        'pick_correct_choice_index',
        'pick_correct_choice_with_choice_given_before_goal',
        'what_is_the_correct_ending',
    ],
    ("rotten_tomatoes",None):[
        'Reviewer Opinion bad good choices',
        'Sentiment with choices'
    ]
}
