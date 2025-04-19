# Grammar-Scoring-Engine using Audio Samples

## Project Overview
Evaluating grammar in spoken English presents unique challenges due to disfluencies, spontaneous phrasing, and non-native variations. This project aims to develop a Grammar Scoring Engine that assesses spoken English proficiency on a scale from 0 to 5. The system integrates:

- OpenAI Whisper for speech-to-text transcription
- DistilBERT for deep NLP-based regression
- Traditional machine learning models (Random Forest, LightGBM, Ridge Regression)
- A meta-ensemble model that combines predictions from both deep learning and traditional models

## Dataset
- **Training Set**: 444 audio clips (45-60 seconds each) with human-assigned grammar scores
- **Test Set**: 195 audio clips without labels

## Pipeline Overview
1. **Audio Preprocessing**: Resampling to 16 kHz mono, trimming silence, normalizing volume
2. **Transcription**: Using OpenAI Whisper for speech-to-text conversion
3. **Transcript Cleaning**: Removing fillers, correcting spacing/punctuation, lowercase conversion
4. **Feature Engineering**:
   - Grammar Error Detection using language_tool_python
   - Syntactic Features (sentence length, POS tag diversity) using spaCy
   - Grammar Correction Edits using T5 model
   - Error Ratios (errors/edits per word)
5. **Model Training**:
   - Fine-tuning DistilBERT for regression
   - Training traditional models (Random Forest, LightGBM, Ridge Regression)
   - Meta-Ensemble with Linear Regression
6. **Evaluation**: MAE, RMSE, Pearson Correlation
7. **Prediction**: Generating final grammar scores for test set

## Model Performance Comparison

| Model                                | MAE  | RMSE | Pearson Correlation |
|--------------------------------------|------|------|---------------------|
| Random Forest Regression             | 0.999| 1.183| 0.285               |
| Ensemble Regression Model            | 1.000| 1.121| 0.320               |
| Ensemble with Combined Features      | 0.985| 1.115| 0.313               |
| DistilBERT Regression                | 0.968| 1.115| 0.313               |
| **Final Meta-Ensemble Model**        | 0.935| 1.073| 0.394               |

The Final Meta-Ensemble Model demonstrates the best performance across all evaluation metrics.

##  Submission.csv Preview ( First 10 rows)

![image](https://github.com/user-attachments/assets/588cf2b1-1084-4c0c-b006-0f77d0d2fcf5)


## Key Insights
- **ASR Quality**: OpenAI Whisper provided robust transcriptions even with noise and accents
- **Feature Engineering**: Syntactic and grammatical features enhanced assessment capability
- **Model Ensemble**: Combining predictions improved performance by leveraging different strengths

## Future Work
- Incorporate prosodic features (pauses, speech rate, intonation)
- Explore ordinal regression models
- Develop end-to-end models working directly from audio
- Extend to multilingual support

## References

- Radford, A., et al. (2022). "Robust Speech Recognition via Large-Scale Weak Supervision." arXiv:2212.04356 [Whisper paper]
- Sanh, V., et al. (2019). "DistilBERT, a distilled version of BERT." arXiv:1910.01108
- Bryant, C., et al. (2019). "Automatic Grammatical Error Correction." Computational Linguistics
