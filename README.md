# LLM Fine-Tuning with LoRA

**MI LAB · B조 땅콩 · 유연, 최진아**

Llama-3-8B를 SAMSum 대화 요약 데이터셋으로 파인튜닝하고, **LoRA를 적용하는 모듈·rank·alpha가 요약 성능과 학습 파라미터 수에 미치는 영향**을 비교한 프로젝트입니다. LoRA 논문의 적용 위치 비교 실험을 출발점으로, V+O 조합의 rank 확장, MLP의 `down_proj` 적용, alpha 조절까지 총 13개 실험을 수행했습니다.

[발표자료 보기](resources/presentation.pdf) · [원본 실험 결과 CSV](resources/experiment_results.csv)

## 연구 질문

- 제한된 학습 파라미터로 대화 요약 성능을 높이려면 어떤 projection에 LoRA를 적용해야 할까?
- 좋은 모듈 조합에서 rank를 늘리면 성능도 계속 개선될까?
- MLP 확장과 alpha 조절은 파라미터 효율에 어떤 영향을 줄까?

LoRA는 사전 학습된 가중치 `W₀`를 고정하고 작은 행렬 `A`, `B`만 학습합니다. 본 프로젝트의 설정에서는 업데이트가 `ΔW = (α/r)BA`이며, 하나의 projection에 추가되는 파라미터 수는 `r × (입력 차원 + 출력 차원)`입니다.

## 모델과 데이터

| 항목 | 설정 |
|---|---|
| Base model | [`meta-llama/Meta-Llama-3-8B`](https://huggingface.co/meta-llama/Meta-Llama-3-8B), base 모델 |
| Dataset | [`knkarthick/samsum`](https://huggingface.co/datasets/knkarthick/samsum) |
| Task | 영어 대화 요약 |
| Train / Validation / Test | 14,732 / 818 / 819개 |
| 학습 방법 | FP16 SFT + LoRA, 원본 가중치 고정 |

학습 입력은 아래 형식이며, 평가 입력에는 정답 요약을 넣지 않습니다.

```text
### Instruction
Summarize the following dialogue.

### Context
{dialogue}

### Answer
{summary}
```

## 공통 학습·평가 설정

| 항목 | 값 |
|---|---|
| Optimizer / Scheduler | AdamW / Linear |
| Learning rate / Warmup ratio | 2e-4 / 0.1 |
| Epochs / Weight decay | 2 / 0 |
| GPU당 학습 배치 / Gradient accumulation | 1 / 64 |
| 유효 학습 배치 | 64, 단일 GPU 기준 |
| LoRA dropout / Bias | 0.1 / none |
| 학습 최대 길이 | 지시문 + 대화 + 정답을 합쳐 512토큰 |
| 평가 입력 최대 길이 / 새 토큰 최대 수 | 512 / 128 |
| 평가 배치 / 생성 방식 | 4 / greedy (`do_sample=False`, `num_beams=1`) |
| Validation / Checkpoint | 100 step마다, 최종 어댑터는 2 epoch 종료 후 저장 |

원본 코드는 입력 전체를 텍스트로 포맷팅해 학습합니다. 공개 코드도 전체 시퀀스에 대한 loss를 사용하며, 정답 부분만 학습하는 completion-only 설정으로 바꾸지 않았습니다.

평가 지표는 ROUGE-1(단어 겹침), ROUGE-2(연속 두 단어 겹침), ROUGE-L(LCS), ROUGE-Lsum(요약 단위 LCS), BERTScore F1(의미적 유사도)입니다. ROUGE는 `use_stemmer=True`, BERTScore는 `roberta-large`, 영어, `idf=True`로 계산합니다. ROUGE-Lsum을 위해 문장별 줄바꿈을 추가하는 별도 후처리는 사용하지 않습니다.

## 실험 결과

**아래 값은 기존 실험의 저장된 점수이며, 수정된 공개 평가 코드로 다시 측정한 결과가 아닙니다.** 원본 평가의 패딩 처리 오류가 확인되어 재평가가 필요합니다. 발표자료의 일부 오기는 ZIP의 점수 CSV와 어댑터 설정을 기준으로 바로잡았습니다. Base 값은 발표자료·엑셀에서 가져왔습니다.

점수는 0–100 스케일이며, CSV에는 0–1 스케일의 값을 저장합니다. `D`는 MLP의 `down_proj`입니다.

| ID | 모듈 | r | α | Params | R-1 | R-2 | R-L | R-Lsum | BERTScore F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base | — | — | — | — | 26.58 | 10.57 | 20.81 | 22.21 | 81.20 |
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
| 13* | V, D | 4 | 8 | 3.01M | 52.91 | 28.86 | 44.54 | 44.54 | 90.95 |

Experiment1과 6은 당초 설계보다 alpha가 크게 설정된 실험으로, 발표에서는 추가 scale 비교로 다룹니다. 학습에 실패한 실험이라는 뜻은 아닙니다.

**\* Experiment13은 검증 보류입니다.** 아카이브의 예측 CSV가 Base 예측 CSV와 바이트 단위로 일치하지만, 점수 CSV에는 위 수치가 기록돼 있습니다. 원인은 미확정이며, MLP 확장에 관한 결론의 근거에서는 제외합니다.

### 결과에서 관찰한 경향

- **적용 위치:** 단일 모듈에서는 V가 적은 파라미터로 높은 점수를 보였습니다. K 단독은 이번 설정에서 상대적으로 점수가 낮았습니다.
- **Rank:** V+O의 rank를 4→16으로 늘리면 파라미터는 4배가 되지만, ROUGE 개선은 작고 BERTScore는 소폭 하락했습니다.
- **Alpha:** 같은 Q,K,V,O·r=2에서 α를 4→16으로 바꾸면 저장된 ROUGE-1이 52.06→53.12로 높아졌습니다. Q,V·r=4에서도 α=16이 α=8보다 점수가 높았습니다.
- **효율:** V 단독과 V+O·r=4는 재평가 후보로 삼을 만한 효율적인 설정입니다. 최고 ROUGE는 Experiment1, 최고 BERTScore는 Experiment4입니다.

이는 Llama-3-8B·SAMSum·이번 하이퍼파라미터에서 관찰한 결과입니다. 여러 seed로 반복하거나 신뢰구간을 계산하지 않아, 작은 차이의 유의성과 다른 과제에 대한 일반화는 확인하지 못했습니다.

### 원논문과 비교할 때의 주의점

원논문의 적용 위치 비교는 GPT-3에서 학습 파라미터 예산을 동일하게 맞춥니다. 이 모델의 저장된 LoRA 행렬에서는 Q/O 출력 차원이 4096, K/V는 1024입니다. 따라서 같은 rank여도 학습 파라미터 수가 다릅니다. 본 실험은 원논문의 비교 방식을 참고했으며, 조건을 완전히 재현한 실험은 아닙니다.

## 레포지토리 구성

```text
llm-lora-samsum/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── experiment.py       # 13개 실험 설정과 공통 프롬프트
│   ├── train.py            # LoRA SFT 학습
│   └── eval.py             # Base/LoRA 추론, ROUGE·BERTScore 평가
└── resources/
    ├── presentation.pdf    # 원본 발표자료
    └── experiment_results.csv  # 기존 점수·파라미터·학습 시간
```

원본 `yeon_lora.py`를 `train.py`로, `yeon_test.py`·`yeon_bert_test.py`·`yeon_basemodel.py`를 공통 `eval.py`로 정리했습니다. 학습 결과는 실행 시 `run/`에 저장되며 Git 추적 대상에서 제외합니다. 모델 가중치·체크포인트·원본 ZIP·데이터셋은 포함하지 않습니다.

## 실행 방법

### 1. 환경과 모델 접근 권한

Python 3.10 이상과 CUDA를 지원하는 NVIDIA GPU가 필요합니다. 원본 실험은 V100을 고려한 FP16 설정입니다. 8B 모델의 FP16 가중치만으로 약 16GB가 필요하며, 학습 시에는 activation 등의 추가 메모리가 필요합니다. 실제 필요 VRAM은 환경에 따라 달라집니다. 양자화는 사용하지 않습니다.

CUDA 환경에 맞는 PyTorch 2.3.1을 [공식 안내](https://pytorch.org/get-started/previous-versions/#v231)에 따라 설치한 뒤, 의존 패키지를 설치합니다.

```bash
python3 -m pip install -r requirements.txt
hf auth login
```

Hugging Face에서 Llama-3 이용 조건에 동의하고, 모델에 접근 가능한 계정으로 로그인하세요. 토큰은 코드에 작성하지 않습니다. 원본에 하드코딩된 토큰은 업로드 코드에서 제거했습니다. 기존 토큰은 소유자가 폐기·재발급해야 합니다.

`requirements.txt`의 주요 프레임워크는 저장된 모델카드의 버전에 맞췄습니다. 보조 패키지의 원본 환경은 모두 기록돼 있지 않아, 전체 환경을 완전히 고정한 파일은 아닙니다.

### 2. 학습

`src/train.py`의 `main()`에서 `experiment = "Experiment10"`을 실행하려는 실험 ID로 변경하세요. 설정 목록은 `src/experiment.py`에 있습니다.

```bash
python3 src/train.py
```

`run/Experiment10/`에 중간 체크포인트, TensorBoard 로그, `trainer_state.json`, 실험 메타데이터, `final_adapter/`가 저장됩니다. 기존 run이 있으면 덮어쓰지 않고 중단합니다. 재실행하려면 기존 run을 다른 위치로 이동하세요.

### 3. 평가

`src/eval.py`의 `main()`에서 동일한 실험 ID를 선택하세요. 베이스 모델은 `experiment = "Base"`로 평가합니다.

```bash
python3 src/eval.py
```

같은 생성 결과로 ROUGE와 BERTScore를 계산하고, `run/<실험명>/evaluation/`에 다음 파일을 저장합니다.

- `predictions.csv`: SAMSum 원본 ID, 예측, 정답
- `scores.csv`: ROUGE 4개 지표와 BERTScore Precision/Recall/F1
- `evaluation.json`: 모델·실험 설정, 생성 조건, BERTScore hash, 예측 CSV의 SHA-256

평가 폴더가 이미 있으면 덮어쓰지 않고 중단합니다. BERTScore 계산 전에는 8B 모델을 해제해 GPU 메모리를 확보합니다.

기존 아카이브의 어댑터를 재평가하려면 대상 실험의 `final_adapter/`를 `run/<실험명>/final_adapter/`에 배치하세요. 예: `workspace/outputs/Experiment10/final_adapter/` → `run/Experiment10/final_adapter/`. 새로 학습할 필요는 없습니다.

## 업로드를 위한 수정과 검증 상태

- 원본 코드의 Hugging Face 토큰을 제거했습니다.
- 생성 시 왼쪽 padding을 사용하며, 결과를 **패딩을 포함한 입력 너비**로 잘라냅니다. 원본은 샘플별 유효 토큰 수로 잘라 예측에 입력 끝부분이 섞였습니다.
- ROUGE와 BERTScore를 같은 예측으로 계산해, 평가 스크립트 사이의 예측 파일 덮어쓰기를 방지합니다.
- 13개 실험 설정을 저장된 어댑터와 대조하고, 학습 시작 전에도 seed를 설정했습니다.
- 원본의 입력 절단 정책(512토큰을 넘는 오른쪽 부분 제거)은 유지합니다. 긴 대화에서는 끝부분이나 `### Answer`가 잘릴 수 있어, 향후 전처리 개선이 필요합니다.

업로드 준비 과정에서 Python 문법, 설정·결과 일치, 생성문 추출, 비밀정보·대용량 파일 제외를 확인했습니다. **정리한 코드의 GPU 학습·추론 및 점수 재측정은 아직 수행하지 않았습니다.** 수정 후 점수는 기존 결과와 달라질 수 있습니다.

## 후속 연구

1. 수정한 코드로 Base와 모든 실험을 재평가하고, Experiment13의 예측과 점수를 대조합니다.
2. 여러 seed로 반복해 점수 차이의 신뢰구간을 확인합니다.
3. 학습 파라미터 예산을 맞춘 적용 위치 비교를 수행합니다.
4. 다른 대화 요약 데이터셋에서 V의 효과를 검증합니다.

## 참고자료와 이용 조건

- Hu et al., [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685), 특히 7.1절·7.2절.
- [Hugging Face PEFT](https://github.com/huggingface/peft)
- [TRL SFTConfig v0.27.0](https://github.com/huggingface/trl/blob/v0.27.0/trl/trainer/sft_config.py)
- [BERTScore](https://github.com/Tiiiger/bert_score)

모델과 데이터셋은 각각 배포처의 이용 조건을 따릅니다. 이 레포지토리는 모델 가중치와 데이터셋을 재배포하지 않습니다.
