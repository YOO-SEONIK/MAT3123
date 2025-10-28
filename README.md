# MAT3123
MAT3123-Final Project

기계학습과응용 기계공학과 2022145155 유선익

Final Project의 목표는 속이 빈 고무 패킹의 열–기계 거동을 물리식으로 라벨링해 합성 데이터셋을 만들고, 이를 학습한 MLP서로게이트로 온도 변동 시나리오에서 응력 기반 $ε-N$ 피로수명을 빠르게 예측한다.

<img width="277" height="273" alt="image" src="https://github.com/user-attachments/assets/fad3d487-e3ae-4f06-bfe0-bf240ead570f" />


1. Data Setting (1~5)
2. Machine Learning (6~9)
3. PyTorch Surrogate Training (10)
4. Sensitivity Analysis (11~12)
5. Thermal-stress Simulation and Fatigue Assessment (13~15)
6. Life Prediction (16~17)
7. Risk-Life Scenario Map (18~19)
19개의 코드셀로 이루어져 있다.

# Data Setting
## 1. 노트북 환경 체크 + 그래프 기본 설정
Python, Numpy, Pandas의 버전을 확인하고 그래프의 설정값을 조절한다.

<img width="201" height="61" alt="image" src="https://github.com/user-attachments/assets/67fb54dc-04cb-4a12-b26b-d4bf22aa68a5" />

## 2. 내부 대류 열전달 계수 h
이 단계에서는 관 내 유동의 내부 대류 열전달계수 $h$를 계산해 이후 온도장 및 응력 해석의 경계조건으로 사용한다. 

먼저 $20\text{–}80~^\circ\mathrm{C}$ 범위에서 물의 평균 물성치(열전도도 $k$, 점도 $\mu$, 밀도 $\rho$, 비열 $c_p$, Prandtl 수 $\mathrm{Pr}$)를 정의한다. 

유속 $U$와 관내경 $D_i$로 Reynolds 수를 $\mathrm{Re}=\dfrac{{\rho}UD_i}{\mu}$ 로 계산하고, 유동 상태를 판별한다. 

층류 $(\mathrm{Re}<2300)$ 에서는 완전발달 내부유동 가정으로 $\mathrm{Nu}=3.66$ 을 사용한다. 

난류 $(\mathrm{Re}\ge 2300)$ 에서는 Dittus–Boelter 상관식 $\mathrm{Nu}=0.023\mathrm{Re}^{0.8}\mathrm{Pr}^{n}$ 을 쓰며, 가열은 $n=0.4$, 냉각은 $n=0.3$ 을 사용한다. 

마지막으로 $h=\dfrac{\mathrm{Nu}k}{D_i}$ 로 환산한다. 

해당 근사는 원형 관, 매끈한 벽, 충분히 긴 관, 상수 물성 등의 단순화를 포함하므로, 고온/점도 변화가 큰 경우나 비원형 덕트, 입구영향이 큰 짧은 관에서는 보정 또는 다른 상관식 검토가 필요하다.

## 3. 고무 링의 온도 분포 T(r)
이 단계는 속 빈 원통(고무 링)의 반지름 방향 정상상태 온도분포 $T(r)$를 구한다. 

축대칭, 축방향 열흐름 무시, 고체 내부 생성열을 $0$, 열전도도 $k$ 일정 가정에서 지배식은 $\dfrac{\mathrm{d}}{\mathrm{d}r}\left(r\dfrac{\mathrm{d}T}{\mathrm{d}r}\right)=0$ 이고 해는 $T(r)=C1ln(r)+C2$ 이다. 

내부/외부 표면에는 대류 경계가 적용된다: $r=r_i$ 에서 $-k\dfrac{\mathrm{d}T}{\mathrm{d}r}=h_i(T-T_{i,\infty})$, $r=r_o$ 에서 $-k\dfrac{\mathrm{d}T}{\mathrm{d}r}=h_o(T-T_{o,\infty})$.

이를 C1, C2에 대한 선형 2x2 방정식으로 정리해 행렬 $M \cdot [C1  C2]^T = rhs$ 를 만들고, $numpy$ 선형해법으로 $C1$, $C2$ 를 푼다. 

그런 다음 $r_i~r_o$ 사이를 균일 격자로 샘플링해 $T(r)=C1ln(r)+C2$ 값을 계산해 반환한다. 

출력은 반지름 벡터 $r[m]$와 온도 벡터 $T[^\circ\mathrm{C}]$ 이며, 이 분포는 이후 열팽창 변형률 및 응력 근사 계산의 입력으로 사용된다. 

가정상 $k$ 와 물성은 상수이고, 복사나 비선형 효과는 무시한다.

## 4. 온도→물성/열팽창→응력(근사) 라벨러
이 단계는 주어진 운전/재료 조건에서 고무 링의 최대 등가응력과 합격/불합격 라벨을 자동 생성하는 핵심 라벨러다. 

먼저 평균 온도 $T_{avg}$로 온도의존 등가 탄성계수 $E(T)=E_{25}\exp\(-\beta\(T-25))$를 계산한다. 

이어 고무의 선형 열팽창 계수 $\alpha$와 기준온도 $T_{ref}$를 사용해 열팽창 변형률 분포 $ε_{th}(r)=\alpha\(T(r)-T_{ref})$을 얻는다. 

구조 거동은 단순화하여 둘레방향 변형이 거의 $0$으로 억제된다고 가정하고, 열팽창에 역행하는 기계 변형을 $ε_{mech}=-ε_{th}$로 두어 원주방향 응력을 $\sigma_\theta=\dfrac{E}{1-\nu}\cdot\varepsilon_{\mathrm{mech}}$ 로 근사한다. 

이 분포의 절댓값 최대치를 최대 등가응력(보수적 von Mises 근사)으로 사용한다. 

label_one_sample은 다음 순서로 동작한다: 

(1) 유속 $U$, 내경 $D_i$로 내부 대류계수 $h_i$ 계산 → (2) 내부/외부 대류 경계와 고무 열전도율 $k$로 속 빈 원통의 정상상태 온도장 $T(r)$ 해석 → (3) $T_{avg}$로 $E$ 산출 → (4) $ε_{th}(r)$ 계산 → (5) $σ_θ(r)$ 와 그 최대치 max_vm 산출 → (6) 허용응력 $σ_{allow}$와 비교해 pass_flag를 $1$과 $0$으로 부여

반환에는 $r$, $T(r)$, $E$, $ε_{th}(r)$, $σ_θ(r)$, max_vm, pass_flag, $h_i$가 포함되어 이후 데이터셋 생성과 모델 학습에 바로 쓸 수 있다. 

이때, 선형 탄성, 등방 열팽창, $ν≈0.49$, 정상상태 열전달, 접촉/구속 단순화 등을 가정한다.

## 5. 랜덤 데이터셋 생성 + 빠른 확인용 플롯
이 단계는 학습에 사용할 표준화된 테이블 데이터를 자동 생성한다. 

먼저 난수 시드로 재현성을 확보한 뒤, 운전/재료 변수($U$, $T_{in}$, $T_{amb}$, $E_{25}$, $β$, $α$, $h_o$)를 현실 범위에서 균일 샘플링한다. 

각 샘플마다 물리 라벨러를 호출해 최대 등가응력 max_vm(회귀 타깃)과 허용응력 기준 pass(분류 타깃)를 계산하고, 이를 하나의 표(df)에 쌓아 CSV로 저장한다. 

생성된 df.head()와 describe()를 통해 분포와 스케일을 즉시 점검하며, 컬럼은 입력 7개($U$, $T_{in}$, $T_{amb}$, $E_{25}$, $β$, $α$, $h_o$)와 출력 2개(max_vm, pass)로 명확히 분리된다. 

이어 개별 프로파일들의 평균이 아니라 평균 파라미터에서 단일 해석을 수행해, 평균 조건에서의 반지름 방향 온도 $T(r)$와 근사 hoop 응력 분포를 플롯하여 물리적으로 가능한지 시각적으로 검증한다. 

마지막으로 데이터셋 합격률(pass_rate)과 평균 조건의 내부 대류계수 $h_i$, max_vm을 함께 출력해 기준을 잡는다. 

이 산출물은 이후 scikit-learn/PyTorch에 그대로 투입할 수 있는 깔끔한 입력-타깃 테이블이며, 학습/검증 분할, 스케일링, 라벨 균형 점검 등의 후속 단계와 연결된다.

<img width="842" height="210" alt="image" src="https://github.com/user-attachments/assets/2661d44f-8a70-4816-b1bc-b4d6a7563db8" />

<img width="748" height="154" alt="image" src="https://github.com/user-attachments/assets/fec761b5-7bf0-462b-adc8-78f6b51c9e9a" />

<img width="642" height="473" alt="image" src="https://github.com/user-attachments/assets/7a026772-0b46-4ea6-bfb0-cc4238d05688" />

<img width="687" height="473" alt="image" src="https://github.com/user-attachments/assets/11a3cae1-4726-4d06-a6bb-d8dd3e673711" />

<img width="671" height="217" alt="image" src="https://github.com/user-attachments/assets/9379c00c-b23a-4a1c-83cd-c9577114a931" />

# Machine Learning

## 6. CSV → 라벨 점검/백업 → 학습/테스트 분할 → 스케일러 저장
이 단계는 학습 입력과 타깃을 확정하고, 분류 타깃의 불균형 여부를 점검한 뒤 필요 시 보조 라벨을 생성하여 안정적인 분류 학습이 가능하도록 데이터를 준비한다.

먼저 dataset.csv를 읽어 입력 피처 7개($U$, $T_{in}$, $T_{amb}$, $E_{25}$, $β$, $α$, $h_o$)와 회귀 타깃(max_vm), 분류 타깃(pass)을 지정한다.

이어 분류 타깃의 클래스 분포를 점검해 한쪽 클래스만 존재하거나, pass 라벨이 존재하지 않거나, 소수 클래스가 20개 미만이면 실제 분류 실습이 불안정하므로, 보조 라벨(pass_median)을 생성한다. 

보조 라벨은 회귀 타깃 max_vm의 중앙값을 기준 임계치로 두어 max_vm <= median 이면 1, 그 외 0으로 정의해 양/음 비율을 균형에 가깝게 만든다(이상적인 구현은 훈련 세트에서 계산한 중앙값을 임계치로 사용해 데이터 누설을 더욱 줄이는 것) 

보조 라벨이 사용된 경우 use_backup_cls=True로 기록된다. 다음으로 입력 $X$, 회귀 타깃 $y_r$, 분류 타깃 $y_c$를 구성하고, train_test_split(stratify=y_c, test_size=0.2, random_state=42)를 사용해 80/20 비율의 층화 분할을 수행한다. 

층화(stratify)는 학습/테스트 양쪽의 클래스 비율을 동일하게 유지해 불균형 데이터에 따른 평가 편차를 최소화한다. 

마지막으로, 전체 설정과 데이터를 전역 딕셔너리 $REG$에 저장한다. 여기에는 입력과 타깃 이름(X_cols, y_reg_name, y_cls_name), 학습/테스트 세트(X_train, X_test, y_r_train, y_r_test, y_c_train, y_c_test), 그리고 보조 라벨 사용 여부가 포함된다. 

입력 스케일 정규화를 위해 StandardScaler를 학습 세트(X_train)에 적합시켜(fit) 동일 스케일을 유지하도록 하고, 이후 MLP 학습 및 시나리오 해석 단계에서도 공용으로 활용할 수 있도록 REG["x_scaler"]로 저장한다. 

결과적으로 이 단계는 데이터 정합성 점검 → 보조 라벨 생성 → 재현 가능한 층화 분할 → 스케일러 저장까지 한 번에 수행하여, 이후 모델 학습 및 평가의 신뢰성을 높이는 기반을 마련한다.

<img width="509" height="32" alt="image" src="https://github.com/user-attachments/assets/72dcde92-035e-4352-80ac-22af7aa6a049" />

## 7. 회귀 베이스라인 학습/평가
이 단계는 회귀 타깃 max_vm을 예측하기 위한 기본 회귀 모델들을 한 번에 학습하고 비교하여, 이후 심화 모델(MLP 등)의 기준 성능을 설정한다.

먼저 KFold(n_splits=5, shuffle=True, random_state=42)로 교차검증 분할자를 생성한다. 이는 전체 데이터를 5개의 폴드로 나누어 각 폴드가 한 번씩 검증 세트로 사용되도록 하며, shuffle=True와 고정된 시드(42)를 통해 재현성을 보장한다.

다음으로 세 가지 모델 구성을 정의한다.

(1) LinearRegression 파이프라인 — StandardScaler()로 입력을 표준화한 후 선형 회귀를 수행하며, 파이프라인 내부에서 스케일러가 훈련 fold에 대해서만 fit되어 데이터 누설을 방지한다.

(2) Ridge 회귀(GridSearchCV) — 동일한 파이프라인 구조로 구성하되, 하이퍼파라미터 alpha ∈ [0.01, 0.1, 1, 10, 100]을 5-가지의 교차검증으로 탐색하여 최적값을 찾는다.

(3) RandomForestRegressor(GridSearchCV) — 비선형 트리 기반 모델로 별도 스케일링이 필요 없으며, n_estimators, max_depth, min_samples_leaf를 탐색 파라미터로 설정해 그리드 탐색을 수행한다.

GridSearchCV는 기본적으로 내부 교차검증에서 R²가 최대가 되는 설정을 선택하고, 학습 완료 후 자동으로 해당 파라미터로 전체 훈련 세트를 재학습한다.

각 모델은 훈련 세트(X_train, y_r_train)로 학습하고 테스트 세트(X_test, y_r_test)에서 예측을 수행하며, 성능 평가는 RMSE(작을수록 예측 오차가 작음을 의미)와 R²(1에 가까울수록 예측이 실제값에 근접함을 의미)의 두 지표를 사용한다. 

또한 GridSearchCV 객체의 경우 best_params_를 함께 저장해, 어떤 파라미터 조합이 최적으로 선택되었는지 기록한다.

모든 결과는 [모델명, RMSE, R², best_params] 형식으로 reg_results DataFrame에 정리되며, RMSE 기준으로 정렬하여 콘솔에 출력한다.

이 표를 통해 각 회귀 모델의 상대 성능을 비교할 수 있고, 향후 결과 분석에도 활용된다.

마지막으로, RMSE가 가장 낮은 모델을 최고 회귀 베이스라인(best_reg_baseline)으로 확정하고, 해당 모델 객체와 RMSE 값을 전역 딕셔너리 REG에 저장한다.

REG에는 baseline_rmse, reg_results, baseline_models_dict 등이 함께 기록되어 이후 보고서 생성 및 시각화 단계에서 그대로 재사용된다.

결과적으로 이 단계는 선형/정규화형/비선형 회귀모델의 성능을 일괄 비교·기록하여 기준 모델을 확정하는 과정으로, 이후 모델 개선 효과를 정량적으로 검증하기 위한 기준점을 마련한다.

<img width="916" height="93" alt="image" src="https://github.com/user-attachments/assets/5e976b14-f16e-4d35-8004-985a2bddb930" />

## 8. 분류 베이스라인 학습/튜닝/평가 및 임계값(τ*)
이 단계는 합격/불합격(pass/fail) 분류를 위한 기본 분류기들을 한 번에 학습·튜닝·평가하고, 운영 목적에 맞는 예측 확률 임계값(τ*)을 기반으로 확정한다.

먼저 모든 모델은 동일한 교차검증 설정(KFold(n_splits=5, shuffle=True, random_state=42))으로 공정 비교한다. 평가 기본 분할은 이전 단계에서 확정된 X_train/X_test, y_c_train/y_c_test를 사용한다.

모델은 세가지를 사용하며 LogReg: StandardScaler → LogisticRegression(max_iter=200, solver="lbfgs") 파이프라인, 파라미터는 C ∈ {0.1, 1, 5, 10}, class_weight ∈ {None, "balanced"}를 GridSearchCV로 탐색한다.

DT(DecisionTree): max_depth ∈ {None, 4, 6, 10}, min_samples_leaf ∈ {1, 3, 5}, class_weight ∈ {None, "balanced"}를 탐색하고, F(RandomForest): n_estimators ∈ {200, 400}, max_depth ∈ {None, 8, 12}, min_samples_leaf ∈ {1, 3}, class_weight ∈ {None, "balanced"}를 탐색한다.

GridSearchCV의 기본 scoring을 따르므로 내부 CV 선택 기준은 정확도(ACC) 이다. 탐색이 끝나면 각 모델은 해당 최적 파라미터로 전체 훈련 세트에 재학습된다.

테스트 세트에서 ACC, 불균형에 강한 F1, 확률 기반 판별력 ROC-AUC를 기록한다.

확률 추정은 predict_proba가 있으면 양성 클래스(1)의 확률을 사용하고, 없으면 decision_function 출력에 시그모이드를 적용하여 대체한다.

결과는 [model, acc, f1, auc, best_params]로 clf_results 표에 정리하고, F1 → AUC 우선 정렬로 최고 모델을 고른다.

최고 모델에 대해 혼동행렬과 classification_report를 출력해 오탐/미탐 구조를 해석한다(실제 0과 1 각각의 맞춤/오분류 개수)

최고 모델의 테스트 확률로부터 precision_recall_curve를 계산하고, 두 과정 중 하나로 τ*를 정한다.

POLICY="max_f1": F1이 최대가 되는 임계값을 선택하거나 POLICY="recall_at": 목표 재현율(TARGET_REC, 기본 0.98) 이상을 만족하는 구간 중 F1이 최대인 임계값을 선택(만족 구간이 없으면 max_f1로 폴백)하여 최종 τ*와 채택 방법을 REG["tau_star"], REG["tau_policy"]에 저장한다.

REG["clf_results"]: 모델별 ACC/F1/AUC/최적 파라미터 표, REG["best_clf"]: 최고 분류기(그리드 최적 추정기), REG["tau_star"], REG["tau_policy"]: 확정 임계값과 방법을 추출한다.

결과적으로 이 단계는 튜닝된 분류기 간 성능 비교 → 최고 모델 선정 → 운영 목적(재현율 보장 등)에 맞춘 임계값 확정까지 일괄 수행하여, 실제 적용 시 오탐/미탐 트레이드오프를 요구 조건에 맞게 제어할 수 있도록 준비한다.

<img width="1126" height="447" alt="image" src="https://github.com/user-attachments/assets/1cbc710c-eb3f-413c-bb47-d1051c385ece" />

## 9. 랜덤포레스트 중요도 시각화 및 회귀 정합 플롯

이 단계는 회귀·분류 모델이 어떤 입력 변수에 의존하는지를 중요도 막대그래프로 확인하고, 이전 단계에서 선정된 회귀 베이스라인의 예측값과 실측값에 대한 정합을 시각적으로 점검한다.

먼저 ７단계(REG["baseline_models_dict"])에서 최종 추정기를 가져와 feature_importances_를 막대그래프로 표시한다. 이는 각 입력 피처($U$, $T_{in}$, $T_{amb}$, $E_{25}$, $β$, $α$, $h_o$)가 트리 분할 시 불순도 감소(회귀=MSE)에 기여한 상대적 중요도를 의미한다. 값이 클수록 해당 변수의 의사결정 기여도가 크다.

다음으로 8단계 (REG["best_clf"])에 대해 파이프라인/그리드서치를 모두 고려하여 결과를 추출한다.

트리/포레스트 계열이면 feature_importances_로 중요도를 그린다.

로지스틱 회귀 등 계수 기반 모델이면 coef_의 절댓값 평균(멀티클래스 대비)을 중요도로 사용한다.

중요도가 높을수록 해당 피처가 모델의 의사결정에 크게 기여했음을 의미하며, 해당 결과는 설계 변수 조정의 우선순위를 정하거나, 데이터 수집·센서 배치의 중요도를 판단하는 근거가 된다.

중요도 해석 시 해당 중요도는 불순도 기반 상대 지표로, 상관 피처 간 분산(importance split)이나 카디널리티/스케일 효과의 영향받아 가중이 분산될 수 있다. 따라서 변수 영향의 정밀한 검증이 필요할 때는 이후 단계에서 Permutation Importance나 SHAP을 보조적으로 검토하는 것이 바람직하다.

마지막으로 7단계 결과표(REG["reg_results"])에서 RMSE 최우수 회귀 모델을 선택해 테스트셋에 대한 Parity 플롯을 그린다. x축은 실제값(True max_vm [Pa]), y축은 예측값(Pred max_vm [Pa])이며, 대각선(y=x)에 가까울수록 정합이 양호하다.

점들이 대각선 주변에 폭넓게 고르게 분포하면 전 구간에서 예측이 안정적임을 시사한다.

체계적인 곡률(언더/오버 예측), 꼬리 부근의 일방적 치우침, 극값에서의 확산 증가는 비선형성 미모델링, 외삽 구간 예측, 타깃/입력 스케일링 이슈, 이상치 영향 등을 의심할 단서가 된다.

이 플롯은 RMSE나 R² 같은 단일 수치가 놓치기 쉬운 오류 패턴(언더/오버 예측 구간, 극값 왜곡 등)을 시각적으로 드러내며, 이를 통해 모델이 실제 데이터의 분포를 얼마나 충실히 학습했는지를 직관적으로 평가할 수 있다. 또한 이러한 시각적 진단은 이후 단계에서의 모델 교체(예: 더 깊은 트리, 비선형 회귀), 입력 특성 개선(스케일·파생피처 조정), 데이터 전처리(이상치 제거·가중 재조정) 등 구체적인 개선 방향을 설정하는 근거로 활용된다.

<img width="584" height="344" alt="image" src="https://github.com/user-attachments/assets/d1632fce-3094-4112-b36b-b29b325817c8" />

<img width="584" height="344" alt="image" src="https://github.com/user-attachments/assets/45a07b56-0d91-41d7-a7e6-c4da68465ebb" />

<img width="488" height="488" alt="image" src="https://github.com/user-attachments/assets/9b3d906a-f316-41fc-9d23-e76fc7b88e2b" />

# PyTorch Surrogate Training

## 10. MLP 서로게이트(수동 SGD) 학습/평가

이 단계는 PyTorch로 간단한 MLP(2×128 ReLU)구조를 학습해 물리 라벨 max_vm(최대 등가응력)을 빠르게 예측하는 서로게이트 모델(surrogate model)를 만든다.

먼저 입력 변수는 StandardScaler로 표준화(평균 $0$, 분산 $1$) 하고, 타깃은 $\log_{10}$ 변환으로 스케일안정화해 극값의 영향을 완화한다.

이 로그 변환은 매우 큰 응력값으로 인한 손실 오류를 방지하고, MSE 손실 계산 시 큰 값의 오차가 과도하게 반영되는 것을 완화하여 수렴 안정성을 높인다.

학습 구조는 torch.optim을 쓰지 않고 순수 SGD 스텝을 직접 구현해 각 epoch마다 loss.backward()로 기울기를 계산하고, 모든 파라미터에 대해 L2 weight를 더해 적용한 뒤 갱신한다.

이 방식은 라이브러리 의존성을 최소화하면서도 SGD의 동작 원리와 L2 효과 구현하여 학습 메커니즘을 재현한다.

학습은 mini-batch(batch_size=64) 방식으로 수행되며, 손실함수는 로그공간의 MSE이다.

또한 검증 손실(val_mse)을 모니터링하며 조기 종료(patience=80) 를 적용해 과적합을 방지한다.

학습 과정에서 매 epoch의 훈련/검증 손실을 모두 기록하여 학습곡선(train_hist, val_hist)을 그리고, 검증 손실이 최소였던 시점의 파라미터를 best_state로 저장한 뒤 최종 복원한다.

학습 완료 후 예측값 $\hat{y}{log} = \log{10}(\hat{y})$를 원공간으로 역변환($\hat{y} = 10^{\hat{y}_{log}}$)하여 실제 단위[Pa]의 응력 예측값을 얻고, 테스트 세트에 대해 RMSE(단위: Pa)를 계산한다.

또한 예측값과 실제값을 비교하는 정합 플롯을 작성해 모델의 일관성과 예측 경향을 시각적으로 확인한다.

점들이 대각선(y=x) 근처에 고르게 분포하면 모델이 전 영역에서 잘 일반화된 것이며, 곡률, 꼬리 쏠림, 극값 영역의 편차가 크다면 비선형성 미모델링·외삽 오차·데이터 스케일 불일치 가능성을 시사한다.

마지막으로 MLP의 테스트 RMSE를 이전 단계의 베이스라인 RMSE와 비교해 더 우수하면 REG["best_reg"]를 MLP로 갱신하고, 그렇지 않으면 기존 베이스라인 모델을 유지한다.

모델, RMSE, 로그변환 여부(inv_target="log10") 등의 메타데이터를 함께 REG에 저장해 이후 시나리오 단계에서 직접 재활용할 수 있도록 한다.

결과적으로 본 단계는 입력 표준화 → 로그 타깃 변환 → 수동 SGD + L2 → 조기 종료 → 원공간 RMSE 평가의 일련 절차를 통해, 단순하지만 안정적인 학습 루프를 구현하고 모델의 수렴과 물리적 정합성을 동시에 검증하는 파이프라인을 완성한다.

이 MLP 서로게이트는 이후 단계에서 반복적 응력 예측이나 피로 해석 등 고빈도 예측 연산의 가속기(core regressor)로 활용된다.

<img width="390" height="276" alt="image" src="https://github.com/user-attachments/assets/a9f35394-72c6-4533-9ec8-69123f500544" />

<img width="584" height="344" alt="image" src="https://github.com/user-attachments/assets/1ad44b92-0da6-41d2-958b-456789a94243" />

<img width="464" height="464" alt="image" src="https://github.com/user-attachments/assets/71f74636-a33f-4be9-966f-d132286a6194" />

# Sensitivity Analysis

## 11. MLP 민감도/탄력도 진단(입력→예측 영향 분석)

이 단계는 학습된 MLP 서로게이트가 예측하는 로그응력 $y=\log_{10}\sigma$에 대해, 각 입력 피처가 출력에 얼마나 영향을 미치는지를 국소 기울기(gradient) 기반으로 정량 분석한다.

먼저 학습 데이터 중 무작위로 $n=256$개의 샘플을 추출하고, 이를 표준화된 입력 공간(z)에서 모델의 입력으로 설정한다. 이후 자동미분(autograd)을 통해 $\dfrac{\partial(log_{10}σ)}{\partial z}$ 를 계산한다.

여기서 $z$는 표준화된 입력이므로, 이 미분은 스케일 차이를 제거한 상대적 민감도를 의미한다. 

각 피처별 절댓값 평균과 표준편차를 계산하고, 막대그래프로 나타내면 모델이 어떤 변수에 더 강하게 반응하는지를 직관적으로 파악할 수 있다.

다음으로, 표준화 좌표를 실제 입력 단위로 환산하여 $\dfrac{\partial(log_{10}σ)}{\partial x} = \dfrac{1}{std(x)} \cdot \dfrac{\partial(log_{10}σ)}{\partial z}$ 를 구한다.

이는 입력 1 단위 변화가 로그 응력값에 미치는 영향을 의미하며, 물리적 단위로 해석할 수 있는 민감도다.

이후 체인룰을 적용하면 응력 자체의 절대 민감도 $\dfrac{\partial σ}{\partial x} = (ln 10)\sigma \cdot \dfrac{\partial(log_{10}σ)}{\partial x}$ 를 계산할 수 있다.

이 값은 각 설계변수가 1 단위 증가할 때 응력($\sigma$, 단위: Pa)이 얼마나 변하는지를 의미하며, 값이 클수록 실제 응력 수준에 미치는 영향이 크다는 것을 나타낸다.

마지막으로 입력 스케일이나 단위를 제거한 무차원 반응성 지표로 탄력도(elasticity) $\left|\frac{\partial \log_{10}\sigma}{\partial \log_{10} x}\right| = \left|\frac{x}{\sigma}\frac{\partial \sigma}{\partial x}\right|$를 계산한다.

탄력도는 “입력 $x$가 일정 비율(% 단위)로 변할 때, 응력 $\sigma$가 몇 % 변하는가”를 나타내는 비율적 민감도로 단위와 스케일에 무관한 모델의 구조적 민감도를 무차원 비율 형태로 보여준다.

이 모든 지표(표준화 민감도, 절대 민감도, 탄력도)는 무작위로 뽑은 여러 샘플에서 평균±표준편차로 계산해 입력 변수별 국소 분산(불확실성)까지 함께 평가된다.

결과적으로 표준화 민감도는 모델 내부에서의 상대적 피처 영향도를, 절대 민감도는 실제 물리 단위(Pa) 기준의 응력 변화량을, 탄력도는 단위와 스케일 차이를 제거한 퍼센트 반응성을 각각 보여준다.

값이 큰 변수일수록 모델이 해당 입력 변화에 민감하게 반응하며, 실제 물리계에서도 중요한 제어 변수일 가능성이 높다.

이는 향후 운전 제어 우선순위, 센서 정밀도 설정, 설계 변수 조정 방향을 정하는 근거로 활용할 수 있다.

다만, 이 해석은 국소(linearized) 근사에 기반하므로, 입력이 크게 변하거나 학습 데이터의 분포를 벗어나는 영역에서는 결과를 보수적으로 해석해야 한다.

<img width="692" height="368" alt="image" src="https://github.com/user-attachments/assets/1c987b48-f513-4daa-89f8-476cfac9ae3f" />

<img width="704" height="368" alt="image" src="https://github.com/user-attachments/assets/4b35b951-7f2f-497b-8078-36fc09783b84" />

<img width="704" height="368" alt="image" src="https://github.com/user-attachments/assets/25ac5f6e-fa50-4c9c-a5cc-d4da5fe18cfa" />

## 12. 유한차분(FD) vs 자동미분(grad) 일치성 점검

이 단계는 앞선 MLP 민감도/탄력도 계산이 정확하고 스케일 변환과 일관되는지 검증하기위해, 자동미분(gradient)으로 계산한 탄력도와 유한차분(Finite Difference, FD)로 근사한 탄력도를 비교 분석한다.

먼저 학습 입력(X_train)에서 무작위로 일부 샘플($n=64$)을 선택하고, 이를 표준화→역변환하여 원단위 입력 $X_{0}$을 얻는다. 각 입력 피처 $x_j$에 대해 ±0.1%의 미세한 곱셈 변화를 주어 $x_j^{(+)} = x_j(1+\varepsilon), x_j^{(-)} = x_j(1-\varepsilon^{-1})$ 형태의 교란을 적용한다. ($\varepsilon = 10^{-3}$, 즉 0.1%)

이때 모델 출력 $y = \log_{10}\sigma$의 변화를 이용해 중앙차분 근사식을 적용하면, $Elasticity_{FD} = \dfrac{y^+ - y^-}{2Δlog_{10}x} \approx \dfrac{Δlog_{10}\sigma}{Δlog_{10}x}$ 로 유한차분 탄력도를 구할 수 있다. 이는 입력이 0.1% 변화했을 때 로그응력이 얼마나 비율적으로 반응하는지를 나타내는 수치적 근사치이다.


동시에 같은 샘플에 대해 자동미분으로 $\dfrac{\partial (log_{10}σ)}{\partial z}$를 구해 스케일러의 표준편차를 반영하여 $\dfrac{\partial(log_{10}σ)}{\partial x} = \dfrac{1}{std(x)} \cdot \dfrac{\partial(log_{10}σ)}{\partial z}$ 로 변환한다.

해당 값을 이용해 로그–로그 탄력도를 이론적으로 계산하면, $\dfrac{\partial log_{10}σ}{\partial log_{10}x} = (ln 10)x \cdot \dfrac{\partial(log_{10}σ)}{\partial x}$ 이 되어 자동미분 기반의 탄력도가 완성된다.

각 피처별로 FD 기반 탄력도의 평균(FD_mean)과 Grad 기반 탄력도의 평균(Grad_mean), 그리고 두 값의 절대오차(abs.err)와 상대오차(rel.err%)를 계산하여 나란히 출력한다.

상대오차가 전체적으로 수 % 이내로 유지되면, 자동미분 경로, 스케일러 역변환, 체인룰 적용이 모두 올바르게 구현되었음을 의미한다.

반대로 오차가 크다면 표준화 스케일 적용 위치나 축 방향 오류, 로그변환 시 계수 누락, 표준화/원단위 혼용, $\varepsilon$이 너무 커서 비선형 항이 반영된 경우등을 점검해야 한다.

해당 검증을 통과하면 수행한 민감도 및 탄력도 해석이 수학적 일관성과 수치적 정확성을 모두 만족하고, 이후 제어 변수 우선순위 설정, 설계 변수 영향 분석, 센서 정밀도 검증 등 민감도 응용 해석에서 결과의 신뢰도를 보장할 수 있다.

<img width="531" height="241" alt="image" src="https://github.com/user-attachments/assets/0f0b8c71-c342-4660-9109-b0d3216af294" />

# Thermal-stress Simulation and Fatigue Assessment

## 13. 피로 해석 기반 함수: 레인플로우, 온도 의존성 탄성률, 시나리오 생성
이 단계는 고무 링의 피로 손상 계산에 필요한 핵심 유틸리티들을 정의한 부분으로, 시간에 따라 변화하는 응력·온도 조건 하에서 누적 손상을 산정하기 위한 기초 모듈 세트를 구성한다.

먼저 _turning_points(y, $ε$) 함수가 시계열에서 기울기 부호가 바뀌는 극값만 추출해 평탄 구간을 제거하고, $ε$ 값(절대 차 컷오프)을 이용해 작은 진동이나 미세한 노이즈 요동을 제거한다. 이렇게 하면 불필요한 작은 사이클이 레인플로우 단계에서 중복 계산되는 문제를 방지하고, 물리적으로 의미 있는 주기만 남길 수 있다.

rainflow(series, ε) 함수는 ASTM 표준에 기반한 레인플로우 사이클 계수 알고리즘을 구현한다. 턴닝 포인트 배열을 스택(Stack)으로 처리하면서 범위(range), 평균(mean), 주기수(count)를 구하고, count=1.0은 완전주기, count=0.5는 반주기를 의미한다. 결과는 (range, mean, count) 열 구조의 배열로 반환되며, 각 행이 하나의 피로 사이클을 나타낸다.

E_of_T(T, E25, β)는 온도 의존 탄성률을 계산하는 헬퍼 함수로, $E(T)=E_{25}\exp\(-\beta\(T-25))$ 를 사용한다.

온도가 상승하면 탄성률이 감소하므로, 열영향을 반영한 응력–변형률 변환에 직접 사용된다. 이 값은 수치 안정성을 위해 하한값(1 kPa)을 보정해 반환한다.

scenario_noise(...)는 온도 시나리오 생성기로, 주어진 평균 온도 $T_m$을 중심으로 난수 잡음을 추가해 변동하는 온도 입력 시계열 $T_{in}(t)$을 만든다. 

분포는 균등 혹은 정규 중 선택할 수 있으며, smooth_tau_s 매개변수를 통해 1차 지연 필터를 적용하면 완만한 시간 변화를 모사할 수 있다.

함수는 시각 배열 $t$, 온도 시계열 $T_{in}$, 그리고 관련 상수($U$, $T_{amb}$, $E_{25}$, $β$, $α$, $h_o$)를 함께 반환하여 후속 해석 단계와 바로 호환 가능하도록 구성된다.

정리하면 이 단계는 온도 시나리오 생성 → 탄성률 E(T) 계산 →레인플로우 주기 분석으로 이어지는 피로 수명 평가의 전처리 기반을 담당한다.

이 모듈들을 통해 이후 단계에서 응력 시계열 → 변형률 → ε–N 수명 → Miner 손상 합산의 전체 피로 해석 파이프라인을 연결할 수 있으며, 서로게이트가 예측한 응력 결과를 실제 피로 수명 평가로 연계하는 핵심구성요소를 제공한다.

## 14. 시간 시나리오 →  σ(t)·p_pass(t)·risk_rate 산출 (서로게이트+분류 연동)
이 단계는 시간에 따라 변하는 운전 입력 시퀀스 $X(t)$를 받아, 서로게이트 회귀로 각 시점의 최대 등가응력 $σ(t)$를 예측하고, 분류기 출력으로 합격 확률 $p_{pass}(t)$ 을 산출한 뒤, 운영 임계값$\tau^*$ 기준의 리스크율을 계산한다.

먼저 학습 타깃 $log_{10}σ$의 분포에서 0.05 ~ 99.95 분위수를 설정하여, 소프트 역변환(soft clamp)을 통해 로그공간 예측값을 원공간 응력으로 복원한다.

이변환은 하드 클리핑 대신 완만한 포화 함수 형태의 $tanh$를 사용하여 외삽 영역에서도 연속적이고 안정적인 복원을 보장한다.

$z = \dfrac{y_{log}-mid}{0.5span}, y_{soft} = mid + tanh(z)(0.5span - 10^{-3}) , \hat{\sigma} = 10^{y{soft}}$, 여기서 $mid = \dfrac{Y_{min}+Y_{max}}{2}, span = Y_{max} - Y_{min}$ 이다.

run_scenario_and_risk(make_xt_fn, T_points) 함수는 시간 배열 $T_points$에 대해 사용자가 정의한 입력 생성기 make_xt_fn(t)을 호출하여, 각 시점별 입력 벡터 ($U$, $T_{in}$, $T_{amb}$, $E_{25}$, $β$, $α$, $h_o$)를 구성한다. 

회귀기가 PyTorch MLP인 경우, 학습 시 사용한 StandardScaler로 표준화한 후 no_grad 모드에서 $\hat{y}{log} = \log{10}(\hat{\sigma})$ 를 계산하며, sklearn 기반 모델은 .predict(X) 결과를 그대로 사용한다. 타깃이 로그 스케일(REG["inv_target"] == "log10")로 학습된 경우 위 식을 이용해 소프트 역변환을 수행한다.

분류기(best_clf)는 동일한 입력$X$ 에 대해 양성 클래스(1)의 확률 $p_{pass}(t)$를 계산한다. 

predict_proba를 지원하지 않을 경우 decision_function 결과를 시그모이드 변환하여 확률을 추정하며,
수치적 안정성을 위해 다음과 같이 확률 범위를 제한한다.

$p_{pass}(t) = clip(p_{pass}(t), 10^{-6}, 1-10^{-6})$
8단계에서 확정된 임계값 $\tau^{*}$ 에 대해, $p_{pass}(t)$가 임계값보다 작은경우 1, 그렇지 않은경우 0으로 ${risk}_{mask}$를 만들고, 전체 시간 중 위험 상태의 비율을 평균하여 리스크율(risk rate) 로 정의한다.

이는 주어진 운전 시나리오에서 불합격 또는 위험 구간이 차지하는 시간 비율로 해석할 수 있으며, 또한 최소 합격 확률과 함께 REG에 저장되어, 이후 단계에서 성능 및 신뢰성 평가의 근거로 사용된다.

결과적으로 본 단계는 시간 시나리오 입력 → 응력 예측 → 확률 추정 → 리스크율 계산으로 이어지는 일련의 과정을 통해, 운전 조건 변화에 따른 응력 응답과 합격 확률의 변동을 동시에 추적하며, 시스템의 불안정 구간을 정량적으로 진단할 수 있는 기반을 마련한다.

## 15. 변형률 진단: 시나리오 생성 → σ(t)·p_pass(t) → ε(t) 분석(통계/플롯)
이 단계는 실제 운전 중 온도·유량 변동을 모사한 시간 시나리오에서, 서로게이트로 예측한 응력 시계열$σ(t)$과 분류 확률 $p_{pass}(t)$를 함께 계산하고, 온도 의존 탄성률로 변형률 $ε(t)$을 구한 뒤 레인플로우 진폭 분포까지 요약및 시각화한다.

먼저 scenario_noise로 2s간격, 총 6h길이의 유입 온도 시나리오 $T_{in}(t)$를 생성한다​(평균온도 70, 변동폭 20, 1차 지연 스무딩 12s) 이때 기록 시간 record_hours_N는  $\dfrac{len(t_N)\cdot DT_S}{3600.0}$ 으로 환산한다.

그다음 run_scenario_and_risk()로 회귀·분류를 동시 수행하여 $σ(t)$, $p_{pass}(t)$, risk_mask, risk_rate를 얻는다. 코드에서는 USE_RICH_SCENARIO=True일 때 입력에 동조 변동을 부여한다. 이로써 순수 온도 변동뿐 아니라 유량·외부대류 계수의 저주파 흔들림이 응력·합격확률에 미치는 영향을 함께 확인할 수 있다.

온도 의존 탄성률은 $E(T)=E_{25}\exp\(-\beta\(T-25))$로 계산하고(수치 안정 하한 $10^{3}$ 적용), 변형률은 $ε(t) = \dfrac{\sigma(t)}{E(T_{in}(t))}$ 로 구한다. 시각화 에서는 $ε(t)$를 %단위로 환산한다.

레인플로우 분석은 $ε(t)$를 입력으로 하여 사이클 범위-평균-가중치를 산출하고, 진폭$ε_a = \dfrac{range}{2}$ 및 중치(완전주기=1.0, 반주기=0.5)를 사용한다. 이번 간결판에서는 진폭 히스토그램만 제시하며, 상단 구간 낭비를 줄이기 위해 x축 상한을 $min(0.08, max ε_a [%])$ 로 제한하고, 세밀한 분포 확인을 위해 bin=1000을 사용한다.

콘솔 요약에는 Risk: 리스크율(%)과 최소 합격확률, 임계값 / Stress: $σ(t)$의 최소/최대/표준편차(단위: Pa) / Strain: $ε(t)$ 평균±표준편차(%) 를 출력한다.

플롯은 4종으로 온도 시계열 $T_{in}(t)$, 응력 시계열 $σ(t)$, 변형률 시계열 $ε(t)$, 레인플로우 진폭 히스토그램를 출력한다.
	​
결과적으로 본 단계는 (시나리오) $T_{in}(t)$, $U(t)$, $h_o(t)$ → (예측) $σ(t)$, $p_{pass}(t)$ → 변환 → $ε(t)$ → (주기분석) $ε_a$의 흐름을 통해, 시간영역 응답·리스크·사이클 진폭 분포를 한 번에 점검하는 정량적 변형률 진단 루틴을 제공한다. 이 요약과 플롯들은 이후 피로 한계 설정(진폭 컷), Miner 손상 누적, 수명 추정의 실무 기준값을 잡는 데 직접 활용된다.



	​

=range/2 및 가중치(완전주기=1.0, 반주기=0.5)를 사용한다. 이번 간결판에서는 진폭 히스토그램만 제시하며, 상단 구간 낭비를 줄이기 위해 x축 상한을 
min
⁡
(
0.08
,
 
max
⁡
𝜀
𝑎
[
%
]
)
min(0.08, maxε
a
	​

[%])로 제한하고, 세밀한 분포 확인을 위해 bin=1000을 사용한다.














이 단계는 실제 운전 중의 온도 변동이 있을 때, 고무 링에 발생하는 변형률 거동을 정량적으로 진단하는 과정이다.

먼저 scenario_noise로 2초 간격, 총 6시간짜리 유입 온도 시나리오 $T_{in}(t)$ 시나리오를 만들고(평균 $80 ^\circ\mathrm{C}$, $±5 ^\circ\mathrm{C}$, 시간상수 $60s$), 이전 단계에서 학습한 MLP 서로게이트로 각 시점의 등가응력 $\sigma(t)$를 예측한다. 

기록 길이 record_hours_N는  $\dfrac{len(t_N)\cdot DT_S}{3600.0}$ 로 계산해 시간 단위로 환산하고, 이후 수명 외삽의 기준으로 사용한다. 

다음으로 온도의존 탄성계수 $E(T)=E_{25}\exp\(-\beta\(T-25))$를 이용해 각 시점의 변형률 $ε(t) = \dfrac{\sigma}{E(T)}$를 계산하고, 이를 %단위로 환산해 시간 변화에 따른 응력-변형률 응답을 얻는다.

계산된 변형률 시계열을 rainflow 알고리즘에 입력해 사이클별 진폭 $ε_a$ 와 카운트(0.5 또는 1.0)을 구한 뒤, 진폭 분포의 요약 통(min, median, max, p1, p05, p10, p90, p95)를 산출해 변형률의 전체 분포 폭과 지배 진폭 수준을 수치로 확인한다.

특히 피로한계 후보 $ε_e$를 바로 잡을 수 있도록 (1) conservative(하위 10% 커트 - 보수적 컷), (2) typical(중앙값의 50% - 대표적 운전 조건), (3) loose(중앙값의 25%- 느슨한 기준)로 세 가지 대표값을 함께 제시하고 이후 $Miner$ 피로 손상 계산 시 진폭 컷오프 설정의 참고치로 활용된다.

마지막으로 (1) 온도 시나리오 플롯 - 시간에 따른 $T_{in}(t)$ 변동, (2) 변형률 시계열 플롯 -시간에 따른 $ε(t)$ [%] 변화, (3)rainflow 진폭 히스토그램 - 진폭 분포 $ε_a$ [%] (가중치=사이클 수) 의 세가지 플롯을 제공해 전체 경향을 직관적으로 확인한다.    

히스토그램 x축은 실제 데이터 상단 구간 낭비를 줄이기 위해 0 ~ $min(0.04, max(ε_a))$로 제한하고, 미세한 분포를 보려 $bins$를 1000개로 설정한다.

결과적으로 해당 단계는 온도–응력–변형률의 시간적 연계, 사이클 분포 통계, 피로한계 후보 도출을 한 번에 수행하는 정량적 변형률 진단 루틴으로, 이후 피로 수명 해석의 입력 조건을 현실적으로 보정하는 데 사용된다.

<img width="700" height="111" alt="image" src="https://github.com/user-attachments/assets/cc0ffc99-1e31-4e6f-928b-2385f6693b9e" />

<img width="704" height="464" alt="image" src="https://github.com/user-attachments/assets/e4b9e279-57a7-4a88-a71d-437e17ffa6d1" />

<img width="704" height="464" alt="image" src="https://github.com/user-attachments/assets/4d9d7dc8-b974-4d63-a0b4-f814760ae583" />

<img width="703" height="464" alt="image" src="https://github.com/user-attachments/assets/3d085652-9694-4e54-b23e-3fde684cc9da" />

# Life Prediction

## 16. ε–N 수명 계산 함수: sigma(t) → ε(t) → 레인플로우 → Miner 손상 → 수명[h]

이 단계는 시간축 $t$와 온도 $T_{in}(t)$, 서로게이트가 예측한 응력 $\sigma(t)$ , 재료 상수($E_{25}$, $\beta$)를 받아 $ε–N$ 모델로 누적손상 $D$와 예상 수명 life_hours를 계산한다. 

먼저 온도의존 탄성계수 $E(T)=E_{25}\exp\(-\beta\(T-25))$로 순간 변형률 시계열 $ε(t) = \dfrac{\sigma(t)}{E(T_{in}(t))}$로 만든다.

그다음으로 변형률 파형을 rainflow에 넣어 사이클별 진폭 $ε_a$와 개수 $count$를 얻고, 진폭을 $ε_a = \dfrac{range}{2}$로 정의한 후, 피로한계 $ε_e$이상만 선택하여 고무용 $ε–N$ 관계 $N_f=(\dfrac{ε_0}{ε_a})^b$로 각 사이클의 허용 반복수를 계산한다.

$Miner$ 합산식 $D=\Sigma \dfrac{count}{N_f}$으로 누적 손상을 구하고, 기록 길이를 record_hours = $\dfrac{t_{end}-t_{start}+\Delta t}{3600.0}$ 로 환산하여 예상수명을 life_hours = $\dfrac{record-hours}{max(D, ε)}$ 로 계산한다(분모 0 방지를 위해 작은 하한 사용)

반환값은 life_hours, $D$ ,record_hours, rf, $ε(t)$, $ε_a$, $count$ 로 수명 및 손상과 함께 중간 결과(레인플로우 결과/변형률 시계열·진폭/사이클 수)까지 돌려주어 파라미터 $ε_0$, $b$, $ε_e$ 튜닝과 결과 해석에 바로 활용할 수 있다.

## 17. ε–N 기반 손상/수명 계산(요약 실행)

이 단계는 앞에서 준비한 시나리오($t_N$, $T_{in,N}$, $const_N$), 서로게이트 응력 시계열 $\sigma_N(t)$ , 그리고 이전에 정의한 함수 compute_life_eN_from_sigma에 넣어 누적손상 $D$와 예상 수명 life_hours를 한 번에 산출한다. 

입력 파라미터(피로 특성) $ε_0$(기준 변형률), $b$(지수), $ε_e$ (피로한계 진폭)은 고무 피로 특성 및 수명에 직접적인 영향이 크므로 설계 가정에 따라 맞게 설정해야한다.

함수 내부에서는 $E(T)$로부터 변형률 $ε(t)$을 만들고, rainflow로 사이클 진폭 $ε_a$ 및 $count$를 계산한 뒤, $ε–N$ 관계 $N_f=(\dfrac{ε_0}{ε_a})^b$와 $Miner$ 합산으로 $D$를 계산한다. 

출력 로그는 (1) $\sigma_N$의 분포 요약(min, max, std), (2)기록 길이 record_hours, (3) 손상 $D$와 수명 life_hours를 보여준다. 

$D$ 약 1이 설계 한계, life_hours는 record_hours를 $D$로 나눈 결과이며, $ε_e$를 높이면 미세 사이클이 무시되어 수명이 길어지고, 반대로 낮추면 더 보수적인 결과가 나온다.

<img width="318" height="67" alt="image" src="https://github.com/user-attachments/assets/4841da15-bccb-4497-8be3-c96839b7e208" />

<img width="698" height="46" alt="image" src="https://github.com/user-attachments/assets/2a882089-b43b-472a-aa4c-dfaa67f1278d" />

