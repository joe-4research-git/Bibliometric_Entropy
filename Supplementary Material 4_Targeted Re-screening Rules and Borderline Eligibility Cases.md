# Supplementary Methods: Targeted Re-screening and Borderline Eligibility Cases

## 1. Purpose and workflow

The two targeted re-screening rules address different types of false negatives revealed by the dual-reviewer audits. Path A addresses missed language evidence among the 6,501 records excluded by automated screening. Path B addresses missed uses of entropy in language research within the 5,015 records that passed automated screening and proceeded to manual assessment. A rule match flagged a record for reassessment; eligibility was determined by reviewer judgment.

The final corpus comprised **735 publications**: 557 initially included, 37 added after the two audits (6 in Path A and 31 in Path B), and 141 added after targeted re-screening (61 in Path A and 80 in Path B). The path totals were 67 for Path A and 668 for Path B. Agreement and Cohen’s κ below are based on the two reviewers’ initial decisions, before final adjudication.

## 2. Path A: Records excluded by automated screening

**What the audit revealed.** The two reviewers reassessed 200 automated exclusions and identified six eligible records. These records had been tagged as lacking a language signal, yet they contained concrete language evidence in their titles or abstracts. For example, S034 paired maximum entropy with information extraction from free text, and S075 linked Shannon entropy to Filipino SMS in the abstract. The recurring problem was that the original rule did not recognize some language objects or tasks as evidence of relevance. The targeted rule addressed this by searching for a broader set of concrete language cues near entropy expressions, including cues appearing only in the abstract.

**Written rule.** The automated exclusion pool contained 6,501 records. Targeted re-screening was limited to the subset originally tagged as having no language signal. A record was flagged if a language expression and an entropy-related expression were no more than 175 characters apart within the title, a single abstract sentence, two adjacent abstract sentences, or the title combined with the first abstract sentence. For evidence outside the title alone, a cue suggesting use (e.g., compute, estimate, train or evaluate) had to occur within 100 characters of the entropy expression. Evidence windows with explicit prior-work wording were skipped unless they also contained an authorship cue (e.g., “we”, “our”, or “this study”). A concrete language cue (e.g., text, lexical, speech, corpus, machine translation) had to appear in the evidence window or in the title or author keywords. Record-wide co-occurrence alone was outside the 478-record reassessment set.

**Results.** In the audit, the reviewers agreed on 197 of 200 records (98.50%; Cohen’s κ = 0.793); 6 were added and 194 remained excluded. In targeted reassessment, they agreed on 435 of 478 records (91.00%; κ = 0.684); 61 were added and 417 remained excluded.

```text
LOCAL_MATCH(record):
    windows = [title] + abstract_sentences + adjacent_abstract_sentence_pairs
    if an abstract sentence exists: append(title + first_abstract_sentence)
    for window in windows:
        if window is not title alone:
            if prior_work_cue(window) and not authorship_cue(window): continue
        for measure in entropy_expression_matches(window):
            if window is not title alone and not use_cue_within_100_chars(measure):
                continue
            for language in language_expression_matches(window):
                if gap_in_characters(language, measure) <= 175
                   and concrete_language_cue(window OR title_and_author_keywords):
                    return first matching window and measure
    return no local match

for record in records_excluded_by_automated_screening:
    if original_exclusion != "no_language_signal": continue
    if in_path_A_audit(record) or in_initial_557(record): continue
    if LOCAL_MATCH(record): flag for dual-reviewer reassessment
```

## 3. Path B: Records retained by automated screening

**What the audit revealed.** The dual-reviewer audit covered 400 of the 5,015 records (200 initially included and 200 initially excluded). The reviewers agreed on 366 of 400 records (91.50%; Cohen’s κ = 0.830). Of the 200 initial exclusions, 31 were added and 169 remained excluded. Some missed studies described the contribution in terms of a task, such as keyword extraction or sentiment analysis, while the substantive use of entropy appeared in the description of the method. For example, R570 used Shannon entropy to extract keywords from literary corpora, and R058 used a generalized cross-entropy approach for sentiment analysis of tweets. These examples motivated a strict rule looking for a direct textual link between a language task and an entropy-related method. The audit also found cases whose evidence was less explicit: the rule was designed to identify a specific pattern of missed records, not to recover every audit false negative.

**Written rule and reassessment set.** The parent population was the 5,015 records retained after automated screening. The targeted search did not reconsider the 557 initial inclusions or the records already covered by the dual-reviewer audit. The first pass looked for language and entropy evidence in the title and abstract, also using author keywords for supporting language cues. It identified local links between language and entropy and retained weaker record-level matches for possible review. For local matches, the procedure looked for cues that entropy was used for measurement or analysis, modeling or training, or evaluation. Matches lacking such cues, and cross-entropy mentions lacking cues of adaptation, comparison or other substantive methodological use, could be downgraded to the lower-priority tier.

The final Path B reassessment set combined two nonoverlapping groups:

| Route | Reassessed | Added | Remained excluded |
|---|---:|---:|---:|
| Strong-signal group: both first-pass candidate tiers (164 priority and 30 lower-priority records) | 194 | 40 | 154 |
| Weak-signal group: strict matches within the first-pass candidate list | 804 | 40 | 764 |
| **Path B total** | **998** | **80** | **918** |

For the second route, the stricter check required an explicit language task or object and an entropy-related expression in the same title with linking wording (e.g., "using", "based", "via"), or in the same abstract sentence with a cue of use. In the sentence route, prior-work wording required an authorship or proposal cue; cross-entropy wording required an additional cue such as adaptation, weighting, comparison or evaluation. Adjacent-sentence and record-level co-occurrence alone did not qualify for this route. Of the 804 records, 96 qualified via the title and 708 via an abstract sentence. The stricter check was applied to the preselected weak-signal candidates rather than directly to the full parent population. The two routes were combined into one Path B worksheet.

```text
candidates = []
for record in records_retained_by_automated_screening:
    if in_initial_557(record) or in_path_B_audit(record): continue
    tier = FIRST_PASS_EVIDENCE_TIER(record)
    if tier not in {priority, lower_priority}: continue
    if strong_signal_group(record):
        candidates.append(record)
    elif STRICT_ASSESSMENT_MATCH(record):
        candidates.append(record)

STRICT_ASSESSMENT_MATCH(record):
    title_pass = language_task(title) and entropy_term(title) and linking_cue(title)
    sentence_pass = any abstract sentence s satisfying all of:
        language_task(s) and entropy_term(s) and use_cue(s)
        not prior_work_cue(s) or authorship_or_proposal_cue(s)
        not cross_entropy(s) or qualifying_cross_entropy_cue(s)
    return title_pass or sentence_pass
```

**Results.** Both reviewers independently reassessed all 998 Path B records. Before adjudication, they agreed on 951 of 998 records (95.29%; Cohen’s κ = 0.746). After adjudication, 80 publications were added and 918 remained excluded. Figure 1 and Supplementary Material 3 report this full reassessment set.

## 4. Four borderline cases with discordant initial decisions

The decisions below are transcribed from the updated reassessment workbook. The explanations identify the logged rule trigger and summarize the available title and abstract evidence. They do not attribute reasons to either reviewer. Both Path B examples illustrate the stricter route within the 998-record reassessment set.

| Case and record | Source and initial decisions | Rule interaction and rationale for the recorded final decision |
|---|---|---|
| **A140. Include** — *Entropy and long-range memory in random symbolic additive Markov chains* (WOS:000378874100004; DOI: 10.1103/PhysRevE.93.062144) | `Path A`. Rater 1: Include. Rater 2: Exclude. **Final: Include.** | The title describes a general symbolic-sequence method. The abstract triggered the rule by placing numerical evaluation of entropy and written English texts in the same sentence. The authors actually applied the conditional-entropy estimation method to written English. DNA was a second application; it did not negate the direct language application. |
| **A120. Exclude** — *Evaluating the Role of Race in Sentencing: An Entropy Weighting Analysis* (WOS:000476935100004; DOI: 10.1080/07418825.2017.1415368) | `Path A`. Rater 1: Exclude. Rater 2: Include. **Final: Exclude.** | The abstract mentions entropy weighting and incarceration sentences in the same sentence, with a comparison cue. The rule treated “sentences” as a language signal. Here the term refers to court penalties. Entropy weighting is used to estimate racial disparities in incarceration; no linguistic object or task is analyzed. |
| **B205. Include** — *Vietnamese treebank construction and entropy-based error detection* (WOS:000358361300001; DOI: 10.1007/s10579-015-9308-5) | `Path B`. Rater 1: Exclude. Rater 2: Include. **Final: Include.** | The strict rule flagged the sentence reporting both error detection and a reduction in corpus entropy after correction. The abstract separately specifies that the method uses conditional entropy to rank candidate annotation errors. The entropy measure therefore plays a direct role in a language task: detecting inconsistencies in a Vietnamese treebank. |
| **B647. Exclude** — *Transitive Entropy—A Rank Ordered Approach for Natural Sequences* (WOS:000528666500008; DOI: 10.1109/JSTSP.2019.2939998) | `Path B`. Rater 1: Exclude. Rater 2: Include. **Final: Exclude.** | The strict rule flagged a sentence describing the proposed entropy measure and its application to pause lengths in speech. The abstract confirms actual entropy use. The borderline issue is scope: the demonstrated outcome was dementia classification, with pause lengths used as clinical predictors. This is an evidence-based explanation of the scope distinction; the workbook does not state the adjudicators’ reason for exclusion. |

**Traceability.** The four decisions can be checked by `Record ID` in Supplementary Material 3 (Dual Reviewer Reassessment of Flagged Records). The audit records used to develop the rules are in Supplementary Material 2 (Dual Reviewer Audit of Initial Screening). Figure 1 in the manuscript shows the complete selection workflow.
