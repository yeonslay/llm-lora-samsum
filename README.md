# LoRA Fine-Tuning for Dialogue Summarization

**Llama-3-8B의 대화 요약 성능과 파라미터 효율을 비교한 LoRA 실험 프로젝트**

MI LAB · 2026 Winter · 팀 땅콩 — 유연, 최진아

대규모 언어 모델을 특정 과제에 적응시킬 때, 학습 파라미터를 늘리는 것만큼 **어디에 업데이트를 적용할 것인가**가 중요합니다. 이 프로젝트는 SAMSum 대화 요약 과제에서 LoRA의 적용 모듈, rank, alpha를 변화시킨 13개 실험을 통해 성능과 학습 파라미터 수의 관계를 분석합니다.

[발표자료](resources/presentation.pdf) · [전체 실험 데이터](resources/experiment_results.csv) · [학습 코드](src/train.py) · [평가 코드](src/eval.py)

## 주요 결과

프로젝트 당시의 평가 기록에서 다음 경향을 관찰했습니다.

| 비교 관점 | 관찰 결과 |
|---|---|
| 적용 모듈 | V 단독은 131만 개의 학습 파라미터로 ROUGE-1 **52.52**, BERTScore F1 **91.18**을 기록했습니다. 단일 모듈 중 ROUGE-1·ROUGE-L·BERTScore가 가장 높았습니다. |
| Rank 확장 | V+O의 rank를 4에서 16으로 늘리면 파라미터 수는 170만 → 682만 개가 되지만, ROUGE-1은 **52.93 → 53.00**으로 소폭 변화했습니다. |
| Alpha 조절 | Q,K,V,O·rank=2에서 alpha를 4에서 16으로 높인 설정은 같은 파라미터 수로 ROUGE-1 **52.06 → 53.12**, ROUGE-2 **28.16 → 29.27**을 기록했습니다. |

이 결과는 큰 rank를 선택하기 전에 적용 위치와 업데이트 스케일을 함께 검토할 필요가 있다는 실험적 시사점을 제공합니다. 위 수치는 재평가 전의 기존 기록이며, 결과의 해석 범위는 아래에 설명합니다.

## 프로젝트 개요

### 연구 목표

LoRA 원논문의 적용 위치 비교를 바탕으로 다음 세 가지 질문을 검토했습니다.

1. 대화 요약에서 어떤 attention projection이 효율적인가?
2. 성능이 좋은 모듈 조합에서 rank 증가의 효과는 어느 정도인가?
3. MLP 확장과 alpha 조절은 추가 파라미터 대비 이점을 제공하는가?

LoRA는 사전 학습 가중치 `W₀`를 고정하고 저차원 행렬 `A`, `B`를 학습합니다. 본 실험의 가중치 업데이트는 다음과 같습니다.

```text
W = W₀ + (α / r)BA
학습 파라미터 수 = r × (입력 차원 + 출력 차원), projection 하나 기준
```

### 수행 내용

- Hugging Face Transformers·PEFT·TRL을 이용한 Llama-3-8B의 LoRA SFT 학습
- Attention 및 MLP projection, rank, alpha를 비교하는 13개 실험 설계·수행
- ROUGE와 BERTScore를 이용한 요약 품질의 정량 평가
- 학습 파라미터 수와 성능을 함께 비교한 효율 분석

## 실험 설계

### 모델과 데이터

| 항목 | 설정 |
|---|---|
| 모델 | [Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B), base 모델 |
| 데이터셋 | [SAMSum](https://huggingface.co/datasets/knkarthick/samsum) |
| 과제 | 영어 대화 → 요약문 생성 |
| Train / Validation / Test | 14,732 / 818 / 819개 |
| 학습 방식 | FP16 SFT + LoRA |

### 비교 구성

| 실험군 | 비교 대상 | 목적 |
|---|---|---|
| 적용 위치 | Q, K, V, O 단독 및 Q+K, Q+V, Q+K+V+O | Attention 내부의 적용 위치 비교 |
| 조합 확장 | V+O, rank 4·8·16 | Rank 증가에 따른 성능과 파라미터 변화 분석 |
| MLP 확장 | V+Down | Feed-forward projection 적용 효과 탐색 |
| 스케일 조절 | Q+V, Q+K+V+O의 alpha 변화 | 파라미터 수를 유지한 업데이트 스케일 비교 |

Q/K/V/O는 각각 `q_proj`/`k_proj`/`v_proj`/`o_proj`, Down은 `down_proj`입니다. 이 모델의 Q/O 출력 차원은 4096, K/V는 1024이므로 같은 rank여도 학습 파라미터 수가 다릅니다. 원논문의 동일 파라미터 예산 실험과는 이 조건에서 차이가 있습니다.

### 공통 설정

| 항목 | 값 |
|---|---|
| Optimizer / Scheduler | AdamW / Linear |
| Learning rate / Warmup ratio | 2e-4 / 0.1 |
| Epochs / Weight decay | 2 / 0 |
| 유효 학습 배치 | 64: GPU당 배치 1 × gradient accumulation 64 |
| LoRA dropout / Bias | 0.1 / none |
| 학습 최대 길이 | 지시문·대화·정답을 합쳐 512토큰 |
| 평가 입력 / 생성 최대 길이 | 512 / 128토큰 |
| 생성 설정 | Greedy decoding, 평가 배치 4 |
| Validation / Checkpoint | 100 step마다 |

학습은 아래 형식의 전체 시퀀스에 대한 loss를 사용합니다. 평가 시에는 `### Answer` 뒤의 정답 요약을 제외합니다.

```text
### Instruction
Summarize the following dialogue.

### Context
{dialogue}

### Answer
{summary}
```

ROUGE는 `use_stemmer=True`로 계산합니다. BERTScore는 `roberta-large`, 영어, `idf=True`를 사용하며 F1을 보고합니다. 아래 점수는 0–100 스케일입니다.

## 실험 결과

### 대표 설정 비교

| 설정 | Rank / Alpha | 학습 파라미터 | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore F1 |
|---|---|---:|---:|---:|---:|---:|
| Base | — | — | 26.58 | 10.57 | 20.81 | 81.20 |
| V | 8 / 16 | 1.31M | 52.52 | 28.09 | 44.21 | **91.18** |
| V+O | 4 / 8 | 1.70M | 52.93 | 28.73 | 44.34 | 91.15 |
| V+O | 16 / 32 | 6.82M | 53.00 | 28.55 | 44.51 | 91.01 |
| Q+K+V+O | 2 / 16 | 1.70M | **53.12** | **29.27** | **44.79** | 90.79 |

<details>
<summary>전체 실험 결과</summary>

| ID | 모듈 | r | α | Params | R-1 | R-2 | R-L | R-Lsum | BERTScore F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | Q, K, V, O | 2 | 16 | 1.70M | 53.12 | 29.27 | 44.79 | 44.76 | 90.79 |
| 2 | Q | 8 | 16 | 2.10M | 51.55 | 27.84 | 43.43 | 43.40 | 90.18 |
| 3 | K | 8 | 16 | 1.31M | 47.79 | 24.18 | 39.42 | 39.43 | 89.43 |
| 4 | V | 8 | 16 | 1.31M | 52.52 | 28.09 | 44.21 | 44.17 | 91.18 |
| 5 | O | 8 | 16 | 2.10M | 52.42 | 28.12 | 43.94 | 43.97 | 90.42 |
| 6 | Q, V | 4 | 16 | 1.70M | 52.35 | 28.25 | 43.85 | 43.90 | 91.03 |
| 7 | Q, K | 4 | 8 | 1.70M | 50.85 | 27.29 | 42.78 | 42.77 | 89.96 |
| 8 | Q, K, V, O | 2 | 4 | 1.70M | 52.06 | 28.16 | 43.71 | 43.69 | 90.84 |
| 9 | Q, V | 4 | 8 | 1.70M | 52.16 | 28.11 | 43.48 | 43.50 | 90.55 |
| 10 | V, O | 4 | 8 | 1.70M | 52.93 | 28.73 | 44.34 | 44.34 | 91.15 |
| 11 | V, O | 8 | 16 | 3.41M | 53.06 | 28.71 | 44.49 | 44.46 | 91.10 |
| 12 | V, O | 16 | 32 | 6.82M | 53.00 | 28.55 | 44.51 | 44.44 | 91.01 |
| 13 | V, Down | 4 | 8 | 3.01M | — | — | — | — | 검증 보류 |

원시 점수, 정확한 파라미터 수, 학습 시간은 [실험 데이터 CSV](resources/experiment_results.csv)에 포함되어 있습니다.

</details>

### 해석과 한계

V 단독은 Q/O 단독보다 적은 파라미터로 높은 ROUGE-1과 BERTScore를 보였습니다. V+O의 rank 확장에서는 파라미터 증가에 비해 점수 변화가 작았고, alpha 조절은 추가 파라미터 없이 ROUGE를 개선한 설정을 보여줬습니다. 이러한 관찰은 적용 모듈과 스케일을 함께 탐색하는 실험 전략을 뒷받침합니다.

결과표는 프로젝트 당시의 평가 기록입니다. 저장소의 평가 코드는 배치 padding과 생성문 추출을 수정한 버전으로, 수정 후 점수는 아직 재측정하지 않았습니다. Experiment13은 예측 파일과 점수 기록의 일치 여부가 확인되지 않아 비교에서 제외했습니다. 단일 모델·데이터셋의 실험이며 여러 seed의 반복 측정이 없어, 작은 점수 차이와 다른 과제로의 일반화에는 추가 검증이 필요합니다.

## 코드 구성

```text
src/
├── experiment.py    # 13개 LoRA 설정과 공통 입력 형식
├── train.py         # SFT 학습 및 어댑터 저장
└── eval.py          # Base/LoRA 추론, ROUGE·BERTScore 계산
resources/
├── presentation.pdf
└── experiment_results.csv
```

학습 결과는 `run/<실험명>/`에 저장합니다. 평가에서는 동일한 예측으로 두 지표를 계산하고, 점수·생성 설정·예측 파일의 SHA-256을 함께 기록합니다. 기존 학습 및 평가 디렉토리가 있으면 덮어쓰지 않고 중단합니다. 모델 가중치와 데이터셋은 저장소에 포함하지 않습니다.

## 실행 방법

Python 3.10 이상, CUDA GPU, Llama-3 모델 접근 권한이 필요합니다. CUDA 환경에 맞는 PyTorch 2.3.1을 [설치](https://pytorch.org/get-started/previous-versions/#v231)한 뒤 다음 명령을 실행합니다.

```bash
python3 -m pip install -r requirements.txt
hf auth login
```

주요 라이브러리 버전은 기존 실험의 모델카드에 맞췄습니다. FP16 학습을 사용하므로 모델 가중치 외에도 activation과 학습 상태를 위한 GPU 메모리가 필요합니다.

**학습:** `src/train.py`의 `main()`에서 실험 ID를 선택합니다. 기본값은 `Experiment10`입니다.

```bash
python3 src/train.py
```

**평가:** `src/eval.py`에서 같은 실험 ID를 선택합니다. 베이스 모델은 `Base`로 설정합니다.

```bash
python3 src/eval.py
```

평가 결과는 `run/<실험명>/evaluation/`의 `predictions.csv`, `scores.csv`, `evaluation.json`에 저장됩니다. 기존 어댑터를 평가하려면 `run/<실험명>/final_adapter/`에 배치합니다.

## 향후 연구

- 수정된 평가 코드로 전체 실험을 재평가하고 여러 seed에서 결과의 안정성 확인
- 학습 파라미터 예산을 맞춘 projection 비교
- 다른 대화 요약 데이터셋을 통한 적용 위치의 효과 검증

## 참고자료

- Hu et al., [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [Hugging Face PEFT](https://github.com/huggingface/peft) · [TRL](https://github.com/huggingface/trl)
- [BERTScore](https://github.com/Tiiiger/bert_score)

모델과 데이터셋은 각 배포처의 이용 조건을 따릅니다.
