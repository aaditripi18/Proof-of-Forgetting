# Submitting the paper to arXiv

Everything arXiv needs is in `arxiv_submission.zip`: the LaTeX source
(`main.tex`, with the references inside it) and the five figures as PDFs.
arXiv compiles it and shows you the PDF before anything is published.

## Before you submit (about 15 minutes)

1. **Make the GitHub repository public.** The paper links to
   https://github.com/aaditripi18/Proof-of-Forgetting, so readers must be able to open it.
2. **Check the references.** Open each one below on Google Scholar and confirm the
   authors, title, venue and year match `main.tex`. The ones marked * are the
   ones most worth a second look.
   - Bourtoule et al. 2021, Machine unlearning (IEEE S&P)
   - Cao and Yang 2015, Towards making systems forget with machine unlearning (IEEE S&P)
   - Carlini et al. 2022, Membership inference attacks from first principles (IEEE S&P)
   - * Gao et al. 2022, VeriFi: Towards verifiable federated unlearning (arXiv:2205.12709)
   - GDPR, Regulation (EU) 2016/679, Article 17
   - Gu et al. 2017, BadNets (arXiv:1708.06733)
   - * Halimi et al. 2022, Federated unlearning: how to efficiently erase a client in FL? (arXiv:2207.05521)
   - * Hayes et al. 2024, Inexact unlearning needs more careful evaluations... (arXiv:2403.01218)
   - He et al. 2016, Deep residual learning (CVPR)
   - Hsu et al. 2019, Measuring the effects of non-identical data distribution... (arXiv:1909.06335)
   - * Kather et al. 2019, Predicting survival from colorectal cancer histology slides... (PLoS Medicine), the PathMNIST source; check the full author list
   - Krizhevsky 2009, Learning multiple layers of features from tiny images (tech report)
   - Kurmanji et al. 2023, Towards unbounded machine unlearning (NeurIPS)
   - * Liu et al. 2021, FedEraser (IWQoS); check the author list
   - McMahan et al. 2017, Communication-efficient learning... (AISTATS)
   - McNemar 1947 (Psychometrika 12(2):153-157)
   - Shokri et al. 2017, Membership inference attacks against ML models (IEEE S&P)
   - * Sommer et al. 2022, Athena: Probabilistic verification of machine unlearning (PoPETs 2022(3))
   - Thudi et al. 2022, On the necessity of auditable algorithmic definitions... (USENIX Security)
   - Wang et al. 2019, Neural Cleanse (IEEE S&P)
   - Wu and He 2018, Group normalization (ECCV)
   - * Yang et al. 2023, MedMNIST v2 (Scientific Data 10:41)
   - Yeom et al. 2018, Privacy risk in machine learning (IEEE CSF)
3. **Optional:** to see the PDF before going to arXiv, upload the zip to
   [Overleaf](https://www.overleaf.com) (free): New Project, Upload Project.
4. **Optional:** replace "Independent researcher" under your name in `main.tex` with
   your university or organisation if you have one, and add an email.

## Submitting

1. Create an account at https://arxiv.org/user/register (use your real name).
2. **Endorsement:** a first-time submitter to a category may need an endorsement
   from someone who has published there. If arXiv asks for one, it gives you a
   code to send to a researcher in machine learning or security (a professor or
   anyone with arXiv papers in cs.LG or cs.CR).
3. Start a new submission:
   - **License:** CC BY 4.0 (lets anyone reuse it with credit).
   - **Archive and category:** Computer Science, primary **cs.LG** (Machine Learning);
     cross-list **cs.CR** (Cryptography and Security).
   - **Files:** upload `arxiv_submission.zip`. arXiv unpacks and compiles it.
   - **Check the PDF preview** arXiv shows: all five figures present, tables
     readable, references at the end.
   - **Metadata:**
     - Title: `Proof of Forgetting? What Membership Tests Can and Cannot Certify in Federated Unlearning`
     - Authors: `Aaditya Tripathi`
     - Abstract: paste the contents of `abstract.txt`
     - Comments: `Code and results: https://github.com/aaditripi18/Proof-of-Forgetting`
4. Submit. Papers submitted before 14:00 US Eastern on a weekday appear the next
   evening (US time).

## Rebuilding the zip

If you edit `main.tex`, rebuild the zip:

    python paper/make_zip.py

If the results ever change, regenerate the figures first:

    python scripts/figures.py --out paper/figures --format pdf
