## Study log
First logged on Github 2026-09-21, prior logs reconstructed.



## 2026-09-21 

Reflecting on my research over the past few days has led me to a realization - if I add a tertiary goal:  "establish efficient baselines for **triage** detecting **small** malignant liver lesions using AI" (small being the operative adjective here), perhaps the answer will become clearer.

So I also ran an evaluation of detection by mass size and epoch budget. The findings there also favor 150 slightly though 750 performs slightly better in terms of overall case-level false positive rate. In this context, the leading models are 150 and 750.

I may choose to add this methodology to the study as it is obviously and in essence implied by the goal of screemning - of course you want to screen malignant masses as small as possible.

Today I will test the saved milestone checkpoints on internal vals data with the following decision matrix in mind:

**Governing metric**: triage-level detection of malignant masses 
**Candidate set**: 150, 300, 750.
**Deciding margin**: I'll choose the smallest epoch budget within 0.02 (accross seeds) of the winning epoch budget


## 2026-09-21  (reconstructed from 2026-09-20)

After further analyzing the results of the pilot epoch budget study and comparing it to similar studies, I continue to delay deciding between a budget of 150 and 300; 750 is also a valid candidate.

At this point, 150 is the strong candidate, with overall triage-level detection of 96%, versus 95% for 300 and 97% for triage. Flagging 1 more mass out of 65 can't justify a 5x compute budget.

Still I want to let the matter settle for a day before deciding.

I considered several options, including testing two runs at  these endpoints and comparing them head to head. I also considered testing to intermediate intervals 200 and 250, so four separate runs in total: 150, 200, 250, and 300, each with three seeds.

Before doing this, however, I decided to test on vals data I already have from the prior saved checkpoints.
. 

## 2026-09-21 (reconstructed from 2026-09-19)

I begin study with a preliminary epoch-budget convergence experiment evaluating various internal milestones saved from a 1000-epoch training run: 50, 100, 150, 300, 500, and 750. 

I intend to use this experiment to find the right epoch budget for this dataset

Analyzing predictions from checkpoints at these milestones against a wide variety of industry-standard detection metrics points to two epoch budgets as representing the ideal budget range: 150 and 300.

I analyze many types of detection but I also analyzed the most important of them (for this study) against mass size.


## 2026-09-21 (reconstructed from 2026-09-18)

After conducting research, I choose to prempt the order of experiments and first run a GPU benchmark, using nnU-Net's own benchmarking tool. I evaluated three different Verda.com GPUs and selected the ideal balance of cost and performance. I chose RTX 6000 Ada.

## 2026-09-21 (reconstructed from 2026-09-17)

After two days of research I have confirmed the ideal GPU hosting provider on which to run this study: Verda.com. I also evaluated Scaleway, gpuyard.com, Exoscale, and Spheron. Evaluation critera were cost, variety of specific GPU models, availability of such models, quality of web application UX (subjective judgement), presence and quality of API (subjective judgement), public ethical commitments to sustainailiby, commitments to data privacy and security, and jurisdiction of company headquarters and hosting locations.

## 2026-09-21 (reconstructed from 2026-09-15)

Once the experiments concerning AUL are complete, I have chosen to use an external validation database. 

After spending a full day researching available databases, I realized there are no other publicly available annotated B-mode ultrasound datasets that contain a mix of normal and abnormal cases.

However, I did find a dataset, LMC-SUD, of B-mode ultrasound abnormal cases. I will test one of my planned detection metrics, triage-level detection against thsi dataset.

## 2026-09-21 (reconstructed from 2026-09-14)

I want cancer screening and monitoring to be low-cost and accessible; my particular interest lies in liver cancer. To that end, this is the goal of this study: establish efficient baselines for detecting malignant liver lesions using AI. 

Baselines doesn't necessarily mean clinical value, but it means research value - a foundation for further studies. Efficiency doesn't just mean compute, though that's part of it. It means low-cost and broadly accessible in every dimension: low-cost to plan, research, design, and also to compute. Post training, it means low-cost and easily reproduced, easy to understand to those without clinical or machine learning expertise, and most importantly, low cost to run. Can it run inference cheaply on consumer hardware?

I will evaluate a broad range of detection metrics with a eye to triage-level flagging yet keep in scope monitoring, as in the use case of cancer patient recurrence monitoring.

For the reasons above, the deep learning network must be open source, relatively simple, and segmentation based. The imaging modality must be ultrasound, as it is by far the lowest-cost and most broadly abvailable. Low quality, B-mode ultrasound, loosely corresponding to cheaper and more portable handheld "POCUS" ultrasound devices is actually preferred over high quality ulstrasound.

Based on those premises, I will conduct the study on U-Net, specifically PlainConvUNet 2D, using nnU-Net to automate the pipeline as much as possible in the interest of efficiency and standardization of baselines. I will use the Annotated Liver Ultrasound (AUL) images dataset as the training corpus.
