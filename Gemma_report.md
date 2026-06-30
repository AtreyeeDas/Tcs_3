# Comparative Architectural Analysis of Cascaded vs. Semi-Multimodal Speech AI Frameworks in Real-Time Cardiovascular Edge Deployments

**Author:** R&I Specialist Engineering Group, Tata Consultancy Services (TCS)

**Target Platform:** Edge Workstation (NVIDIA RTX PRO 5000, Blackwell `sm_120`, PyTorch Nightly `cu124`)

**Task Domain:** Multilingual Real-Time Patient-Doctor Dialogues (Cardiology Scenarios)

---

## Abstract

This report presents a thorough data-driven performance evaluation of two real-time edge speech AI architectures deployed for localized, multilingual cardiovascular patient care. **Pipeline 1** utilizes an explicit cascaded pipeline consisting of *Silero VAD $\rightarrow$ Faster-Whisper (Large-v3) $\rightarrow$ Nemotron-4B-Mini-Hindi $\rightarrow$ Coqui XTTS-v2*. **Pipeline 2** implements a unified, semi-multimodal approach using *Silero VAD $\rightarrow$ Gemma Audio-In Native Core $\rightarrow$ Coqui XTTS-v2*, where automated regex patterns unpack implicit transcriptions and response variables from a single model forward pass.

Based on a test dataset spanning 35 sequential multilingual conversational turns across 7 clinical archetypes, we analyze the systemic tradeoffs between architectural modularity and native multimodal integration. Our findings reveal that while the cascaded model achieves vastly superior processing latencies, it introduces fatal error-propagation vulnerabilities and clinical safety risks. Conversely, the semi-multimodal engine provides remarkable contextual accuracy and safety, but introduces a significant "structural tag blockade" latency penalty.

---

## 1. Experimental Domain & Performance Framework

To evaluate both systems under conditions reflecting real-world clinical stressors, 35 prompts were divided into 7 thematic 5-turn conversations:

* **Conversations 1 & 2:** Everyday baseline diagnostics and medication adherence (English).
* **Conversations 3 & 4:** Blood pressure anxieties and cascading cardiac crises (Hindi/Hinglish).
* **Conversation 5:** Technical cardiology jargon stress-testing (ECG, LVEF, Mitral regurgitation).
* **Conversation 6:** Medical history tracking and clinical pushback handling.
* **Conversation 7:** Acute emergency escalation (Cardiogenic shock/Pulmonary edema simulation).

---

## 2. Quantitative Latency & Throughput Profile

```
      AVERAGE SYSTEM LATENCY & THROUGHPUT COMPARISON
+------------------------------------------+------------+------------+
| Metric Parameter                         | Pipeline 1 | Pipeline 2 |
|                                          | (Cascade)  | (Gemma MM) |
+------------------------------------------+------------+------------+
| ASR Latency / Audio Ingestion (sec)     |   0.385s   |    Merged  |
| LLM Reasoning / Generation Latency (sec) |   4.655s   |    Merged  |
| Unified Processing Latency (UPL) (sec)   |   5.040s   |   8.622s   |
| Time-To-First-Audio (TTFA) (sec)         |   4.672s   |   8.799s   |
| LLM/Gemma Text Throughput (Tokens/sec)   |  24.53 tok |  13.73 tok |
| Audio Ingestion Velocity (Tokens/sec)    | 173.52 tok |  40.82 tok |
| Global Word Error Rate (WER)             |   21.59%   |   16.71%   |
+------------------------------------------+------------+------------+

```

### 2.1 Core Processing Analysis ($UPL_{sec}$)

A side-by-side verification of unified core processing latencies reveals an unexpected hardware paradox on the Blackwell (`sm_120`) architecture. The combined extraction cost of Pipeline 1 ($\text{ASR} + \text{LLM} = 5.040\text{s}$) outperforms Pipeline 2's multimodal pass ($\text{UPL} = 8.622\text{s}$) by **3.582 seconds on average**.

The text-generation throughput metrics confirm this performance gap: Nemotron-4B moves text efficiently at **24.53 tokens/sec**, whereas Gemma’s cross-modal context window throttles generation speed down to **13.73 tokens/sec**. This behavior stems from the compute overhead required by Gemma to cross-attend to heavy raw audio feature embeddings throughout the entire token generation sequence.

### 2.2 Ingestion Rate & Linguistic Scale

A sharp divergence occurs during language-switching phases. In Conversation 3 and 4 (Hindi runs), Pipeline 1's `ASR_Audio_Token_Rate` drops from over **200 tokens/sec** (English baseline) down to a low of **75.82 tokens/sec** (Turn 4.1). Faster-Whisper chokes up when decoding code-switched phrases or mapping phonemes to Devanagari characters.

Gemma’s audio ingestion rate, although globally slower, remains remarkably flat and stable—hovering consistently between **30 and 42 tokens/sec** regardless of whether the patient speaks English, Hindi, or a highly distorted medical Hinglish dialect.

---

## 3. Structural Flaws & Algorithmic Blockades

### 3.1 Pipeline 1: The Cascaded Error-Propagation Vulnerability

The prominent structural flaw observed in the fully modular pipeline is **Acoustic-to-Semantic Error Cascading**. Because individual layers function blindly, acoustic mis-transcriptions generated by Faster-Whisper directly poison the context fed into Nemotron-4B. This breakdown is clearly illustrated in three critical failure modes:

1. **The Stroke Diagnosis Hallucination (Turn 5.1):** The user asks about atrial fibrillation: *"...Does my ECG show any signs of A-Fib?"* Whisper transcribes "ECG" as **"ICP"**. Nemotron reads "ICP" (Intracranial Pressure), ignores the cardiac reference, and hallucinates an irreversible clinical assessment: *"Yes, it's possible that your ischemic stroke case record indicates atrial fibrillation..."*
2. **The Interaction Blindspot (Turn 6.4):** The patient asks if calcium channel blockers interact with *Clopidogrel*. Whisper transcribes the antiplatelet drug as **"clubby dog rhythm"**. Nemotron entirely misses the drug interaction hazard, defining a fictional pathology instead: *"A clubbed dog rhythm is a heart rhythm disturbance that can occur with certain congenital heart conditions..."*
3. **Terminology Corruption (Turn 5.4):** The patient mentions *mitral valve regurgitation*. Whisper corrupts this into **"mild menstrual bulk recarbitation"**. Nemotron completely fails to identify the valvular issue, shifting its reasoning to a myocardial infarction.

### 3.2 Pipeline 2: The Structural Tag-Blockade Phenomenon

Pipeline 2 circumvents lexical corruption by using an integrated multimodal architecture, resulting in a cleaner transcription profile. However, it introduces a severe architectural defect: **The Sync-Tag Output Blockade**.

Because Gemma is engineered to return structured outputs inside explicit XML brackets (`<transcript>`, `<response>`, `<response_lang>`), the orchestrator's regex parser **cannot slice the string until the model hits its final stop token** and closes the `</response>` block. This completely breaks streaming playback capabilities.

Even if a short sentence is completed early in the generation loop, the system forces the audio streaming threads to idle until the entire 3-tag structure finishes executing. This structural limitation explains why Pipeline 2’s average `TTFA_sec` is tightly bound to its global `UPL_sec` (**8.799s**). This latency is critically high for an elderly patient experiencing respiratory distress.

---

## 4. Linguistic, Empathy & Clinical Analysis

### 4.1 Cross-Lingual Failure Modes

The evaluation reveals distinct language-handling limitations in both architectures:

* **Pipeline 1 Incomplete Response Loop:** During Hindi conversational turns (3.1 to 3.5), Pipeline 1 consistently crashes mid-sentence or drops formatting syntax. It truncates critical clinical clauses mid-thought (e.g., Turn 3.2: *"...यदि आपके पास उच्च रक्तचाप का"*). This behavior is driven by severe sub-word token fragmentation occurring inside the English-centric tokenizer when processing Devanagari scripts, exhausting the max token generation budget prematurely.
* **Pipeline 2 Cross-Lingual Hallucination (Turn 6.4):** The patient speaks in clean English regarding *Clopidogrel*. Gemma’s cross-modal encoder experiences an alignment failure, cross-attends erroneously to historical Hindi context windows, and outputs the transcription and response entirely in Devanagari script: *"क्या कैल्शियम चैनल ब्लॉकर्स फॉर द एंजाइना..."*

### 4.2 Clinical Appropriateness & Crisis Escalation

The most stark contrast between the two pipelines lies in emergency protocol handling. Pipeline 1 fails completely as a safe medical assistant during acute cardiovascular crises:

* **The Pulmonary Edema Failure (Turn 7.5):** The patient presents pathognomonic symptoms of crushing chest pain and acute pulmonary edema: *"...my chest feels like someone is standing on it... and I'm coughing up a pink, frothy liquid..."* Nemotron-4B responds with passive, educational text: *"The frothy liquid could be a sign of lung issues, such as pneumonia... Here are some steps you can take to prepare for your medical consultation..."* It completely misses the life-threatening emergency, fails to trigger a high-urgency advisory, and leaves the patient in a dangerous loop.
* **Pipeline 2 Emergency Precision:** Under the exact same emergency scenario (Turn 7.5), Gemma immediately drops all non-essential advice, cuts through generic diagnostic filler, and issues an immediate, clear directive: *"...this is a medical emergency and you must go to the nearest emergency room immediately for urgent evaluation and treatment."*

---

## 5. System Optimization & Strategic Engineering Blueprint

To resolve these architectural limitations before moving toward a production edge environment, two targeted software engineering interventions must be implemented:

```
   OPTIMIZED MULTIMODAL STREAMING ARTIFACT PIPELINE
+-------------------------------------------------------+
|              Incoming Audio Stream                    |
+-------------------------------------------------------+
                           |
                           v
+-------------------------------------------------------+
|                  Silero VAD Layer                     |
+-------------------------------------------------------+
                           |
                           v
+-------------------------------------------------------+
|    Gemma Audio-In Multimodal Core Engine              |
|    -> Generates Token Stream:                         |
|       t_1, t_2: <transcript> ...                      |
|       t_n:      <response>                            |
+-------------------------------------------------------+
                           |
                           v  [Intercepts Token Stream at Output Layer]
+-------------------------------------------------------+
|       Asynchronous Token Token-Streaming Parser        |
|  - Drops '<response>' and historical tags on-the-fly  |
|  - Buffers characters into active words               |
+-------------------------------------------------------+
                           |
                           v  [Yields Completed Sentence Arrays]
+-------------------------------------------------------+
|  Downstream Coqui XTTS Sentence Chunker / Synthesis   |
|  -> Bypasses global execution blockade                |
|  -> Decreases physical TTFA from ~8.7s to <1.2s        |
+-------------------------------------------------------+

```

### 5.1 Mitigation of Cascade Failure: Semantic-Biased Error Injection

To maintain Pipeline 1's low core processing latency while fixing its vulnerability to ASR transcription corruption, future development should implement a **Semantic-Biased Medical Matcher** directly between Faster-Whisper and Nemotron.
Rather than passing raw text strings blindly, the transcript must stream through a localized phonetic Levenshtein distance matcher tuned to a hardcoded dictionary of cardiology entities. If the model records low-confidence terms like *"clubby dog rhythm"* or *"menstrual bulk recarbitation"*, the matcher will intercept the string and perform a semantic correction to *"Clopidogrel"* and *"mitral valve regurgitation"* before the tokens touch the LLM context window.

### 5.2 Mitigation of Multimodal Latency: Asynchronous Token-Streaming Parsing

To fix the high TTFA blockades observed in Pipeline 2, the orchestrator's synchronous tag extraction framework must be replaced with an **Asynchronous Token-Streaming Parser**.

Instead of calling a blocking function that waits for the full text output string, the code must intercept the raw generation token stream byte-by-byte. A lightweight state machine can monitor the token stream: when it encounters the `<transcript>` tag, it suppresses the output; the moment the `<response>` token is emitted, the parser goes live, unblocks downstream queues, strips the tag wrapper on-the-fly, and pushes the text to Coqui XTTS sentence-by-sentence. This optimization will decouple the system's response latency from total generation times, dropping the physical TTFA from **8.79 seconds down to a highly responsive sub-1.5-second conversational pace**.
