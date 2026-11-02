# Submitting the paper to arXiv, step by step

Everything arXiv needs is in `arxiv_submission.zip`: the LaTeX source
(`main.tex`, with the references inside it) and the five figures as PDFs.
arXiv turns it into a PDF and shows it to you before anything is published,
and nothing is public until you press the final Submit button.

Plan on about an hour the first time, plus however long endorsement takes.

---

## Part 1: Before you start (do these first)

1. **Make the GitHub repository public.** The paper links to
   https://github.com/aaditripi18/Proof-of-Forgetting, so readers must be able to open it.
2. **Look at the PDF once (recommended).** Go to https://www.overleaf.com, sign up
   free, click **New Project → Upload Project**, and choose `arxiv_submission.zip`.
   Overleaf shows the finished paper on the right. Read it through once. You can
   also use **Share** there to get a link to send to an endorser.
3. **Check the references** (list at the end of this file). Search each on Google
   Scholar and confirm the authors, title, venue and year match. If one is wrong,
   say so and it will be fixed before you upload.

---

## Part 2: Create your arXiv account

1. Go to **https://arxiv.org/user/register**.
2. Fill in the form:
   - **Email:** your VIT email (you can change it to a personal email later in
     your account settings: do that before you graduate).
   - **Username and password:** your choice.
   - **Name:** Aaditya Tripathi, exactly as on the paper.
   - **Affiliation:** Vellore Institute of Technology
   - **Country:** India
   - **Career status:** Student
   - **Default category:** Computer Science, Machine Learning (cs.LG)
3. Agree to the terms and submit.
4. **Open your VIT inbox** and click the confirmation link arXiv sends. Your
   account works only after this.
5. Optional but useful: make a free **ORCID iD** at https://orcid.org and add it
   to your arXiv profile. It stays yours for life, whatever your email.

---

## Part 3: Submit the paper

Log in at https://arxiv.org/login and click **START NEW SUBMISSION**. arXiv walks
you through a few pages:

### Page 1: Policies and license
- Confirm that you are the author and that you agree to arXiv's policies.
- **License:** choose **CC BY 4.0** (anyone may reuse the paper if they credit you).
- **Archive / category:** Computer Science, primary category **cs.LG (Machine Learning)**.

**If arXiv says you need an endorsement**, this is where it happens. It stops and
shows an **endorsement code** (six characters) and a link to send.
Don't worry: your submission is saved, and you continue from the same place later.
- Send the code to someone who can endorse you (a VIT professor who has posted
  ML or security papers on arXiv, or a researcher in the field; an email template
  is at the end of this file).
- They open the link, enter the code, and you get an email when it's done.
- Then log back in and continue the submission (it is listed under your
  incomplete submissions).

### Page 2: Upload files
- Click **Choose file**, pick `arxiv_submission.zip`, and upload.
- arXiv unpacks it. You should see `main.tex` and five files under `figures/`.

### Page 3: Process
- Click **Process**. arXiv compiles the paper (about a minute).
- Click **View PDF** and check it: title and your name at the top, five figures,
  tables readable, references at the end.
- If it shows an error instead, copy the error message and ask for a fix; do not
  continue with a broken PDF.

### Page 4: Metadata (what readers see on the arXiv page)
- **Title:** `Proof of Forgetting? What Membership Tests Can and Cannot Certify in Federated Unlearning`
- **Authors:** `Aaditya Tripathi`
- **Abstract:** open `abstract.txt`, copy everything, paste.
- **Comments:** `12 pages, 5 figures. Code and results: https://github.com/aaditripi18/Proof-of-Forgetting`
- **Cross-list (additional category):** add **cs.CR (Cryptography and Security)**.
- Leave Journal reference, DOI, MSC and ACM class empty.

### Page 5: Preview and submit
- Check everything once more, then click **Submit**.
- arXiv emails you a submission number (like `submit/1234567`). It is not public yet.

---

## Part 4: After you submit

- arXiv moderators check new papers (a quick check that it is real research in
  the right category; they do not review the science).
- Papers submitted before **14:00 US Eastern time** (about **23:30 India time**)
  on a weekday usually appear the **next day**, around 06:00 India time. Friday
  submissions appear after the weekend.
- When it appears you get an email with your **arXiv number** (like
  `arXiv:2611.01234`) and a public link. Put that link on your resume, LinkedIn,
  portfolio and the GitHub README.
- Need to fix something later? You can upload a new version (v2) at any time; the
  old version stays visible.
- If moderators put the paper "on hold", they email you a reason; reply politely or
  make the change they ask for.

---

## References to check

The ones marked * are the most worth a second look.

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

---

## Endorsement request email (template)

> **Subject:** arXiv endorsement request (cs.LG): verifying federated unlearning
>
> Dear Dr. [Name],
>
> I'm a student at Vellore Institute of Technology and have written a paper on
> whether membership-inference tests can verify unlearning in federated learning.
> It shows that the loss-threshold MIA most papers report cannot tell a server that
> did nothing from a genuine retrain, while client-level LiRA can, but only above a
> client-size threshold that I measure with a controlled experiment.
>
> Paper: [Overleaf share link]
> Code and all results: https://github.com/aaditripi18/Proof-of-Forgetting
>
> This is my first arXiv submission, and I would be grateful if you would consider
> endorsing me for cs.LG. My endorsement code is [CODE]. If you'd rather not, I
> completely understand, and any feedback on the paper would be very welcome.
>
> Thank you for your time,
> Aaditya Tripathi

---

## Rebuilding the zip

If `main.tex` is edited, rebuild the zip before uploading:

    python paper/make_zip.py

If the results ever change, regenerate the figures first:

    python scripts/figures.py --out paper/figures --format pdf
