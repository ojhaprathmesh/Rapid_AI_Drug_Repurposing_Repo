# Elsevier Submission Companion Package
**Target Journal:** *Computers in Biology and Medicine* (Elsevier)  
**Publishing Model:** Hybrid ($0 Author Fee / Traditional Subscription)  
**Manuscript Title:** *Rapid AI Drug Repurposing: A GraphSAGE-based Clinical Discovery Lab for Drug Repurposing on PrimeKG*

---

## 1. Research Highlights (Mandatory for CBM)
*Elsevier requirement: 3 to 5 bullet points, each strictly ≤ 85 characters including spaces.*

1. Inductive GraphSAGE framework for drug repurposing on PrimeKG.
2. Enforces a 4-tier protocol eliminating topological data leakage.
3. Achieves 0.9734 AUC-ROC, 93.51% recall, and 2.50 µs CPU scoring.
4. Hub-penalized path saliency grounds local LLM clinical explanations.

---

## 2. CRediT Authorship Contribution Statement
*To paste into the online form or include in the manuscript.*

* **Prathmesh Ojha:** Conceptualization, Methodology, Software, Formal analysis, Data curation, Writing – review & editing.
* **Jhankar Bansal:** Conceptualization, Methodology, Software, Validation, Data curation, Writing – original draft.
* **Nikhil Kumar:** Supervision, Project administration, Resources, Writing – review.

---

## 3. Declaration of Competing Interest
*Standard statement:*

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

---

## 4. Cover Letter to the Editor-in-Chief

```text
Dear Editor-in-Chief,
Computers in Biology and Medicine, Elsevier

We are pleased to submit our original research article entitled "Rapid AI Drug Repurposing: A GraphSAGE-based Clinical Discovery Lab for Drug Repurposing on PrimeKG" for consideration in Computers in Biology and Medicine.

In this work, we present an end-to-end, privacy-preserving computational framework combining inductive Graph Neural Networks (GraphSAGE) over the Precision Medicine Knowledge Graph (PrimeKG) with on-premise Large Language Model explainability. Our core contributions directly align with the translational scope of Computers in Biology and Medicine:

1. Methodological Rigor: We implement a 4-tier data leakage safeguard eliminating transductive shortcut leakage common in biomedical graph benchmarks.
2. Empirical Performance: Our model achieves an AUC-ROC of 0.9734, an accuracy of 91.97%, and a recall of 93.51% on 1,880 held-out test pairs, accompanied by real-time bedside CPU latency (2.50 µs cached scoring).
3. Explainable AI (XAI): We formulate a degree-penalized path saliency metric that overcomes the ubiquitous hub-protein problem, grounding local LLM generation in verified 2-hop biological pathways.
4. Clinical Validation: We validate top-ranked candidate predictions against independent published clinical trials and mechanistic assays with PubMed PMIDs.

This manuscript has not been published previously and is not under consideration for publication elsewhere. All authors have approved the manuscript for submission.

Thank you for your time and consideration.

Sincerely,
Prathmesh Ojha, Jhankar Bansal, Nikhil Kumar
School of Engineering and Technology, BML Munjal University, Gurugram, India
Corresponding Author: nikhil.kumar@bmu.edu.in
```

---

## 5. Suggested Reviewers (Elsevier Often Requests 3-4 Reviewers)
When asked during submission, you can suggest researchers who have published on PrimeKG or biomedical Graph Neural Networks:
1. **Marinka Zitnik** (Harvard Medical School) — Lead investigator of PrimeKG.
2. **Kexin Huang** (Stanford University) — Expert in Therapeutics Data Commons and GNN drug discovery.
3. **Pietro Liò** (University of Cambridge) — Expert in biomedical Graph Neural Networks.
