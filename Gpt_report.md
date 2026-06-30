Yes. I went through the complete dataset again, not just the responses but also the transcripts, token statistics, WER/CER, latency distribution, TTS metrics and comparative evaluation. Below is a report written in a concise research-paper style. I intentionally avoided simply saying "Pipeline 2 is better" and instead highlighted the underlying patterns observed from the experiments.

---

# 6. Experimental Evaluation and Analysis

## 6.1 Experimental Overview

Two speech-based conversational pipelines were evaluated for a cardio-focused multilingual healthcare assistant.

**Pipeline 1 (Cascaded):**

Silero VAD → Whisper Large-v3 Turbo ASR → Nemotron-4B Hindi Mini Instruct → Coqui XTTS

**Pipeline 2 (End-to-End SpeechLM):**

Silero VAD → Gemma SpeechLM (Unified Speech Understanding + Reasoning) → Coqui XTTS

Evaluation was performed over **35 multi-turn conversations** covering routine consultations, medication counselling, cardiovascular terminology, Hindi and English interactions, emotional conversations and emergency cardiac scenarios. Both objective metrics and detailed subjective evaluation were considered.

---

# 6.2 Automatic Speech Recognition Performance

Overall ASR quality remained strong for routine conversational English, with most common vocabulary transcribed accurately by both systems. However, several important observations emerged for medically relevant conversations.

Medical terminology remains one of the primary challenges. Complex cardiovascular terms such as *mitral valve regurgitation*, *premature ventricular contraction*, *Clopidogrel*, *ACE inhibitors* and *angina pectoris* were frequently distorted by Pipeline 1, often altering the clinical meaning of the user's query. In several cases these transcription errors directly propagated into incorrect downstream reasoning.

Hindi ASR performance exhibited noticeably greater variability than English. Although common conversational Hindi was generally recognized correctly, recognition quality degraded for medical terminology, mixed Hindi-English (Hinglish), and longer spontaneous utterances. Several conversations demonstrated incorrect recognition of symptom names while preserving only numerical values such as blood pressure or heart rate.

Pipeline 2 generally produced lower word error rates than Pipeline 1 (average WER 0.167 vs 0.216), resulting in more reliable downstream reasoning. Nevertheless, occasional language inconsistencies remained, particularly in multilingual conversations where English responses were produced for Hindi inputs or vice versa.

### Key observations

* English conversational speech achieved consistently high recognition accuracy.
* Medical terminology remains considerably more error-prone than everyday vocabulary.
* Hindi and Hinglish continue to present greater recognition challenges.
* Pipeline 2 demonstrated improved robustness to medical vocabulary and spontaneous speech.
* Even small ASR errors can substantially alter subsequent clinical reasoning.

---

# 6.3 Response Quality Analysis

Response quality revealed the largest difference between the two architectures.

Pipeline 1 generally produced factually reasonable responses for straightforward informational queries. However, responses frequently resembled educational articles rather than interactive conversations. Answers often became generic, overly verbose, or failed to address the patient's exact concern. Several responses defaulted to broad lifestyle advice irrespective of the user's actual question.

Pipeline 2 demonstrated significantly stronger contextual reasoning. Responses remained more focused on the user's immediate concern, acknowledged previous conversational context more effectively, and generally avoided introducing unrelated information. Instead of immediately generating explanations, the system frequently requested clinically relevant follow-up information before reaching conclusions.

However, Pipeline 2 also exhibited certain limitations. In multiple multi-turn conversations, previously discussed symptoms were not consistently incorporated into later responses. Although each individual reply remained appropriate, long-term conversational continuity could still be improved.

### Key observations

Pipeline 1

* Frequently produced generic healthcare advice.
* Often expanded beyond the user's question.
* Educational rather than conversational.
* Occasionally failed to answer the exact query.

Pipeline 2

* More focused and context aware.
* Better follow-up questioning.
* Better conversational flow.
* Stronger reasoning with fewer unsupported assumptions.
* Long-term symptom tracking remains incomplete.

---

# 6.4 Empathy and Human-Likeness

One of the clearest improvements observed in Pipeline 2 was conversational quality.

Pipeline 1 generally maintained a polite tone but frequently felt scripted. Responses commonly followed repetitive templates and often overlooked the patient's emotional state before presenting medical information. Hindi conversations repeatedly began with identical greetings, reducing conversational naturalness.

Pipeline 2 consistently acknowledged emotions before providing medical guidance. Expressions such as "I understand that you are worried", "Please do not worry", and "That must be frightening" made conversations resemble real clinician-patient interactions. Emotional validation was particularly effective during medication counselling, anxiety-related conversations and emergency presentations.

Nevertheless, emotional continuity across multiple conversational turns remains an area for improvement. The model occasionally responded only to the latest symptom without explicitly linking it to previously discussed complaints.

### Observed strengths

* Natural acknowledgement of patient emotions.
* Less template-driven dialogue.
* Better reassurance without excessive medical jargon.
* Improved conversational realism.

### Remaining limitations

* Emotional context is not consistently maintained across multiple turns.
* Previous symptoms are sometimes forgotten during longer conversations.

---

# 6.5 Clinical Safety and Medical Reasoning

Clinical safety represents the most important evaluation criterion for healthcare deployment.

Pipeline 1 demonstrated several clinically significant failures.

Examples include:

* recommending double aspirin doses after missed medication,
* suggesting aspirin should be taken on an empty stomach,
* confusing heart rate with blood glucose,
* interpreting mitral valve regurgitation as myocardial infarction,
* failing to recognize emergency cardiac symptoms with sufficient urgency.

Many of these failures originated from ASR errors propagating directly into downstream reasoning.

Pipeline 2 substantially reduced these issues. The model consistently avoided unsupported diagnoses, appropriately deferred interpretation of unavailable investigations, and frequently recommended professional medical evaluation instead of speculative advice.

Emergency recognition also improved considerably. Severe symptoms such as crushing chest pain, pink frothy sputum, orthopnea, severe palpitations and syncope were usually escalated appropriately.

However, several responses still recommended routine clinical appointments where emergency referral could arguably have been more appropriate.

### Overall observations

Pipeline 1

* Higher susceptibility to unsafe recommendations.
* Greater propagation of ASR errors.
* Frequent assumptions without sufficient evidence.

Pipeline 2

* More conservative clinical reasoning.
* Better uncertainty handling.
* Better emergency recognition.
* Safer medication counselling.

---

# 6.6 Multilingual Performance

Both pipelines successfully supported English and Hindi conversations.

However, several multilingual challenges were identified.

Pipeline 1 occasionally distorted Hindi symptom descriptions sufficiently to alter downstream reasoning.

Pipeline 2 generally preserved Hindi semantics more effectively but demonstrated inconsistent language preservation. Multiple Hindi conversations generated English responses despite correctly recognizing Hindi speech. Conversely, one English interaction unexpectedly produced a Hindi response.

These observations indicate that multilingual understanding is stronger than multilingual response control.

### Observations

* Hindi ASR is reasonably accurate but less robust than English.
* Language preservation during response generation remains inconsistent.
* Hinglish and code-switching require further evaluation.

---

# 6.7 Emergency Scenario Handling

Emergency cardiac scenarios included:

* chest pain
* orthopnea
* severe palpitations
* left arm heaviness
* syncope
* pink frothy sputum

Pipeline 1 frequently recognized these symptoms but often concluded with generic advice such as consulting a healthcare provider rather than emphasizing immediate emergency care.

Pipeline 2 demonstrated substantially improved emergency escalation. Life-threatening symptom combinations were generally recognized promptly, with recommendations for urgent hospital evaluation.

Nevertheless, responses often focused only on the latest symptom without integrating earlier warning signs discussed during the conversation.

---

# 6.8 Latency Analysis

Latency measurements indicate a clear trade-off between response quality and processing speed.

Pipeline 1 demonstrated significantly lower unified processing latency, averaging approximately **5.04 seconds**, compared to **8.62 seconds** for Pipeline 2.

Similarly, average Time-To-First-Audio increased from **4.67 seconds** in Pipeline 1 to **8.80 seconds** in Pipeline 2.

The increased latency primarily originates from the larger unified reasoning stage within the end-to-end SpeechLM. In contrast, ASR processing itself contributes only a small fraction of the total processing time.

Interestingly, several emergency conversations in Pipeline 2 exhibited substantially lower latency due to concise responses, suggesting that response length strongly influences overall interaction time.

Overall, latency is dominated by language reasoning rather than speech processing.

---

# 6.9 Speech Synthesis Analysis

Coqui XTTS produced natural and intelligible speech across both pipelines.

Speaking rates remained within a comfortable conversational range, typically between **11–14 characters per second**, supporting clear and understandable communication.

Real-Time Factor values remained close to real-time generation for most conversations, indicating efficient synthesis performance.

Since the same TTS system was used in both pipelines, speech naturalness remained largely consistent.

---

# 6.10 Overall Comparative Analysis

The cascaded architecture demonstrates lower latency and faster interaction, making it attractive for real-time deployment. However, its modular design makes it more vulnerable to error propagation, where transcription mistakes directly influence downstream language reasoning. This effect becomes particularly pronounced for medical terminology and multilingual conversations.

The end-to-end SpeechLM architecture exhibits stronger contextual understanding, improved empathy, safer clinical reasoning, and substantially better handling of emergency situations. The unified reasoning stage appears more robust against imperfect transcripts and produces responses that more closely resemble natural clinician-patient conversations.

Despite these improvements, several challenges remain before practical deployment. Long-term conversational memory requires strengthening so that earlier symptoms continue influencing later reasoning. Language consistency across multilingual interactions must also be improved to ensure that responses consistently match the user's language. Finally, reducing unified processing latency remains essential for responsive real-time clinical interaction on edge devices.

Overall, the experimental results suggest that **response quality in healthcare applications depends more strongly on robust clinical reasoning and conversational understanding than on marginal improvements in transcription accuracy alone**. While the cascaded architecture offers superior computational efficiency, the end-to-end SpeechLM provides significantly stronger conversational capability and clinical reliability, making it a more promising foundation for future multilingual cardiovascular conversational agents, provided that latency and multilingual response consistency are further optimized.
