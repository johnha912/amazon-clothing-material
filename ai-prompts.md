# AI Prompts Used

Every prompt sent to any AI tool for this project, in the order used. The report must list all of them.

| Date | Tool | Prompt | Used for |
|---|---|---|---|
| 2026-10-19 | Claude Code | Create a shared pandas method-chain template for the team. | `pipeline.py` |
| 2026-10-19 | Claude Code | Commit and push the pipeline. | Git |
| 2026-10-19 | Claude Code | Should we use ydata-profiling and share its report on GitHub? | EDA tool choice |
| 2026-10-19 | Claude Code | Shorten this prompt log and write it in English. | This log |
| 2026-10-19 | Claude Code | Commit and push the log update. | Git |
| 2026-10-19 | Claude Code | Should ydata-profiling be used for the EDA task in issue #1? | EDA tool choice |
| 2026-10-19 | Claude Code | Turn the pipeline into a Markdown rules file for teammates' AI tools. | `AGENTS.md` |
| 2026-10-19 | Claude Code | Put the author names in a README table. | `README.md` |
| 2026-10-19 | Claude Code | Leave the README setup steps unchanged. | `README.md` |
| 2026-10-19 | Claude Code | Make teammates' AI tools log each prompt and push it automatically. | `AGENTS.md` |
| 2026-10-21 | Claude Code | Revise AGENTS.md: add agent roles, Task B leakage rules, label rules, shared eval, status, and ownership; shorten code style. | `AGENTS.md`, `docs/code-style.md` |
| 2026-10-27 | Claude Code | Upgrade the agent team in AGENTS.md to cover every report and presentation rubric block, with cross-tool audits and compute routing. | `AGENTS.md` |
| 2026-10-27 | Claude Code | Commit and push the AGENTS.md agent-team upgrade. | Git |
| 2026-10-27 | Claude Code | Move the Kaggle related-implementations work from Rui Song to Nguyen Ha. | `AGENTS.md` |
| 2026-10-27 | Claude Code | Add the missing Modeling role with Yanglin Hu as lead to AGENTS.md. | `AGENTS.md` |
| 2026-10-27 | Claude Code | Check, debug, and push the Issue #1 work: EDA, TF-IDF material models, rating models, model notes, and report drafts. | `analysis/`, `docs/model-notes/`, `report/`, `results/` |
| 2026-10-27 | Claude Code | Re-export the pipeline diagram as a clean image without the editor toolbar. | `report/assets/project-pipeline.png` |
| 2026-10-27 | Claude Code | Rewrite the report sections to be shorter and easier to read, with tables and supporting formulas. | `report/sections/` |
| 2026-10-27 | Claude Code | Restructure the report to the rubric with full paragraph counts, tables, formulas, and figures, and fix the text-cleaning bug that erased titles. | `report/sections/`, `analysis/load_data.py`, `docs/model-notes/` |
| 2026-10-27 | Claude Code | Update the AGENTS.md Status section with the current pilot macro-F1 (0.5572). | `AGENTS.md` |
| 2026-10-27 | Claude Code | Write the Related Implementations section from three Kaggle notebooks on Amazon Reviews 2023. | `report/sections/related_implementations.md` |
| 2026-10-27 | Claude Code | Commit and push the Status update and the Related Implementations section. | Git |
| 2026-10-29 | Claude Code | Pull the latest repo and rewrite the README with the task overview, pilot results, setup, layout, roadmap, and team. | `README.md` |
| 2026-11-02 | Codex | Implement my DistilBERT material-model contribution, save test macro-F1, and prepare comparison and shoe-scope coordination with Nguyen. | `analysis/material_bert.py` |
| 2026-11-02 | Codex | Prepare a Colab-ready script because there is no existing project notebook. | DistilBERT Colab workflow |
| 2026-11-02 | Codex | Explain the Colab missing-file error after cloning the repository and enabling the T4 GPU. | Colab script upload troubleshooting |
| 2026-11-02 | Codex | Diagnose the Colab Trainer import failure caused by the torchvision nms operator and correct the text-model setup instructions. | Colab dependency troubleshooting |
| 2026-11-02 | Codex | Fix the Colab pretrained-model revision check after successful weight downloads and explain how to retry training. | Model revision tracking and Colab retry |
| 2026-11-02 | Codex | Read the completed Colab run and provide the remaining paste-ready cells to inspect, document and save the results. | Post-training Colab cells and model note |
| 2026-11-02 | Codex | Clarify which helper file to choose in the Colab upload dialog. | Post-training helper file selection |
| 2026-11-02 | Codex | Investigate the failed post-training helper cell shown in the Colab screenshot. | Post-training helper troubleshooting |
| 2026-11-02 | Codex | Provide the backup-size TypeError traceback after the model note and prompt log were saved. | Backup download correction |
| 2026-11-02 | Codex | Package all completed work into one ZIP in Downloads and write a PDF brief explaining my contribution, findings and requests for John/Nguyen and Rui. | Final handoff ZIP and PDF brief |
| 2026-11-02 | Codex | Include the original CS6140stage1.ipynb downloaded from Colab in the handoff. | Original Colab notebook inclusion |
| 2026-11-05 | Claude Code | Review the project state from memory and the repo. | Project status |
| 2026-11-05 | Claude Code | Check the GitHub repo and Issue #1 for updates. | Issue #1 |
| 2026-11-05 | Claude Code | Use subagents to find useful, verified papers for the project and list all candidates for review. | `report/sections/related_work.md` |
| 2026-11-05 | Claude Code | Open all reference links so I can check each one myself. | `report/sections/related_work.md` |
| 2026-11-05 | Claude Code | Audit the diff before pushing and fix the issues it found. | `report/sections/related_work.md` |
| 2026-11-08 | Claude Code | Check Issue #1 against everyone's pushed work, integrate Yanglin's DistilBERT handoff, and finish the remaining Nguyen tasks. | Issue #1, `report/`, `README.md`, `AGENTS.md`, `results/` |
| 2026-11-08 | Claude Code | Drop the member column from the prompt log. | `ai-prompts.md`, `AGENTS.md` |
| 2026-11-08 | Claude Code | Replace the HTML and PNG project pipeline with a Mermaid diagram. | `report/assets/project-pipeline.mmd`, `README.md`, `report/sections/04_data_analysis.md` |
| 2026-11-08 | Claude Code | Use Yanglin's release zip as the DistilBERT source of record. | `analysis/material_bert.py`, `results/` |
| 2026-11-08 | Claude Code | Check everything still missing in Issue #1 and finish it. | Issue #1, `report/sections/related_work.md` |
| 2026-11-08 | Claude Code | Commit and push the Related Work section and tick the finished Issue #1 items. | Git, Issue #1 |
| 2026-11-08 | Claude Code | Draft the Week 2 plan (Nov 9–15) to post as a new issue once Issue #1 is closed. | Week 2 issue |
| 2026-11-08 | Claude Code | Post the sample check, close Issue #1, and open the Week 2 issue. | Issues #1 and #2 |
| 2026-11-08 | Claude Code | Check the README and rewrite it where needed. | `README.md` |
| 2026-11-08 | Claude Code | Add the five papers Rui found for the Method and Analysis sections to the repo; the project can draw on them too. | `docs/method-analysis-sources.md`, `README.md` |
| 2026-11-08 | Claude Code | Work these sources into the Method and Analysis sections. | `report/sections/05_methods.md`, `report/sections/08_analysis.md` |
| 2026-11-08 | Claude Code | Merge the per-section reference lists into one shared list. | `report/sections/11_references.md`, `05_methods.md`, `08_analysis.md`, `related_work.md` |
| 2026-11-08 | Claude Code | Fill in the year for the three Kaggle notebooks in the reference list. | `report/sections/11_references.md` |
| 2026-11-08 | Claude Code | Are meeting notes required? If not, delete them and push. | `report/meeting-notes.md`, `README.md` |
| 2026-11-08 | Claude Code | Update the README. | `README.md` |
| 2026-11-14 | Claude Code | Check whether the Issue #2 code package I prepared is OK. | `analysis/material_tfidf.py`, `analysis/rating.py` |
| 2026-11-14 | Claude Code | Go ahead with your plan and use a team of subagents if needed: rerun on the 250k sample and update the report; tell me when it is ready to push. | `results/`, `report/sections/`, `docs/model-notes/`, `README.md`, `AGENTS.md` |
| 2026-11-14 | Claude Code | Prioritize my tasks that do not affect the other two members. | Scope of the Issue #2 changes |
| 2026-11-14 | Claude Code | OK, rerun TF-IDF and rating after the text-cleaning fix. | `analysis/load_data.py`, `results/` |
| 2026-11-14 | Claude Code | Should we move the rating run to Kaggle or Colab? Guide me to run it on my machine. | `results/rating*.csv`, `results/rating_residual_plot.png` |
| 2026-11-16 | Claude Code | Add a run.json provenance file for the TF-IDF and rating runs so teammates can follow them. | `analysis/load_data.py`, `analysis/material_tfidf.py`, `analysis/rating.py`, `results/*_run.json` |
| 2026-11-23 | Claude Code | Audit and push the Task C plan and the AGENTS.md rule so teammates can review them with the Week 3 issue. | `docs/feature-plan.md`, `AGENTS.md`, Issue #3 |
| 2026-11-23 | Claude Code | Tighten the README, then commit and push the finished work. | `README.md`, Task C commit |
| 2026-11-23 | Claude Code | Integrate my Issue #3 package (report sections, Task C charts, CI driver) and leave teammates' files to them. | `report/sections/`, `analysis/task_c_charts.py`, `analysis/ci_from_predictions.py` |
| 2026-11-23 | Claude Code | Rerun Task B on the 250k sample with the rating_number >= 5 noise filter and the shared listing features. | `analysis/rating.py`, `results/rating*` |
| 2026-11-23 | Claude Code | Save the Kaggle SST-2 agreement notebook as a .py script and commit its summary outputs. | `analysis/taskc_sst2_agreement.py`, `results/taskc_sst2_*.csv` |
| 2026-11-23 | Claude Code | Tick the finished steps on Issues #2 and #3. | Issues #2, #3 |
| 2026-11-24 | Claude Code | Check Issues #2 and #3 and the latest repo state before I start. | Issues #2, #3 |
| 2026-11-24 | Claude Code | Pull the repo after backing up my local Related Work and review my parts against the instructor feedback. | `src/evaluate.py`, `app/app.py`, `report/sections/related_work.md` |
| 2026-11-24 | Claude Code | Use three separate agents to check my parts for errors. | `src/`, `app/`, `report/sections/related_work.md` |
| 2026-11-25 | Claude Code | Verify the remaining paper numbers against the sources. | `report/sections/related_work.md` |
| 2026-11-25 | Claude Code | Run the final audit before pushing and fix what it finds. | All changed files |
| 2026-11-25 | Claude Code | Read the repo and do my Week 2 and Week 3 DistilBERT work (Issues #2 and #3). | `analysis/material_bert.py`, `docs/model-notes/material_bert.md`, `results/material_bert*`, `results/material_comparison.csv`, `ai-prompts.md` |
| 2026-11-25 | Claude Code | Ask what is still running. | DistilBERT 250k run status |
| 2026-11-25 | Codex | Audit the uncommitted DistilBERT 250k changes against AGENTS.md, read-only, and report PASS or CHANGES REQUESTED. | Audit of `analysis/material_bert.py`, `docs/model-notes/material_bert.md`, `results/material_bert*`, `ai-prompts.md` |
| 2026-11-25 | Claude Code | Ask whether the work is ready to submit, given the Codex audit (changes requested), and fix the findings. | `analysis/material_bert.py`, `docs/model-notes/material_bert.md`, `results/material_bert*` |
| 2026-11-25 | Codex | Re-audit the fixes from the first audit, read-only, including the two-commit provenance plan and the handoff. | Second audit of the same files and the handoff |
| 2026-11-25 | Claude Code | Fix the second audit's findings without committing or sending anything outside the local machine. | `analysis/material_bert.py`, `docs/model-notes/material_bert.md`, `ai-prompts.md`, proposed `src/evaluate.py` change for Rui |
| 2026-11-25 | Codex | Third audit pass, read-only: check the evaluator patch in an isolated copy, the fiber-word wording, the hyperparameter sources and the push guard. | Third audit of the same files, the handoff and the commit kit |
| 2026-11-25 | Claude Code | Fix the third audit's push-check finding, write a status summary of what is done and what is still needed, and give step-by-step submission instructions without submitting anything. | Commit kit, handoff, `ai-prompts.md` |
| 2026-11-25 | Claude Code | Ask to push the work with only my name on the commits, without an AI co-author line. | Commit messages in the commit kit |
| 2026-11-25 | Claude Code | Push before Rui's evaluator change and hand it to Rui and John afterwards; give me the prompt for the next Codex audit. | `analysis/material_bert.py`, `docs/model-notes/material_bert.md`, commit kit, follow-up patch for Rui |
| 2026-11-25 | Claude Code | Share the fourth-pass Codex prompt as sent (no audit result yet). | Fourth Codex audit |
| 2026-11-25 | Codex | Fourth audit pass, read-only: the push-first plan, the evaluator handoff patches, the note and the push kit. | Fourth audit of the same files and the commit kit |
| 2026-11-25 | Claude Code | Fix the fourth audit's finding (the evaluator change must reach main with the push) without editing Rui's file. | `analysis/material_bert.py`, `docs/model-notes/material_bert.md`, commit kit |
| 2026-11-25 | Claude Code | Push my part first, then message Rui and John with what I pushed and what they need to do; nothing to Rui before the upload. | Commit kit (push-first route and its commit message), report drafts for John, messages to Rui and John |
| 2026-11-25 | Codex | Fifth audit pass, read-only: the push-first plan, its route in the commit kit, and the files from the fourth pass. | Fifth audit of the same files and the commit kit |
| 2026-11-25 | Claude Code | Act on the fifth audit: drop the push-first route (the script must run on main when pushed), check every outgoing commit for changes to src/, and ask Rui before the upload. | Commit kit, report drafts for John, messages to Rui and John |
| 2026-11-25 | Claude Code | Rui approved including his evaluator change in my push; record his approval and prepare the sixth Codex audit. | Commit message for Rui's change, sixth-pass Codex prompt |
| 2026-11-25 | Codex | Sixth audit pass, read-only: Rui-approved evaluator change carried as its own commit, the commit kit, and the files from the fifth pass. | Sixth audit of the same files and the commit kit |
| 2026-11-25 | Claude Code | Ask what is left for me on Issue #3; approve the Task C plan and push the approval. | `docs/feature-plan.md` approval line |
| 2026-11-25 | Codex | Audit, read-only: the Task C approval diff and its commit message. | Task C approval line, commit message |
| 2026-11-25 | Claude Code | Fix the audit finding: the commit message named a Codex row the prompt log did not have yet. | Commit message for the Task C approval |
| 2026-11-25 | Codex | Re-audit, read-only: the Task C approval diff and the corrected commit message. | Task C approval line, commit message |
| 2026-11-25 | Claude Code | Audit the repo after the latest teammate updates and report whether it is ready for the slides. | Repo audit, Issues #2 and #3 |
| 2026-11-25 | Claude Code | Audit which code and pushes are still missing. | Repo audit |
| 2026-11-25 | Claude Code | Fix the bugs found in the audit: add the missing paired and Task C confidence intervals and update the report with the DistilBERT 250k results. | `analysis/ci_from_predictions.py`, `analysis/review_sentiment.py`, `results/`, `report/sections/`, `README.md`, `AGENTS.md` |
| 2026-11-25 | Claude Code | Improve the efficiency of the coding tasks. | Coding workflow |
| 2026-11-25 | Claude Code | Continue the bug fixes. | Same files |
| 2026-11-26 | Claude Code | Commit the audited fixes, trim the report to the 8-page limit, and log the prompts. | Git, `report/sections/`, `ai-prompts.md` |
| 2026-11-26 | Claude Code | Run the demo locally for a check. | `app/app.py` |
| 2026-11-26 | Claude Code | Switch the demo to a calibrated linear SVM and check the report against the proposal's methods. | `analysis/material_tfidf.py`, `app/app.py`, `results/demo_model.csv`, `report/sections/` |
| 2026-11-26 | Claude Code | Compare the new demo model with the old one. | `results/demo_model.csv` |
| 2026-11-26 | Claude Code | Push the work to a separate branch for team review before merging into main. | Git |
