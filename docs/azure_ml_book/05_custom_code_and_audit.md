# 산출물 5 — 커스텀 코드 모듈 설계안 + 추가 작업 감사 (Elston 후속)

> 작성: 2026-06-23. 맥락: Elston 기반 개선(산출물 04) 완료 후 ①life matrix verify 하네스 점검,
> ②2-7 커스텀 코드 모듈 설계, ③추가 작업 감사. 본 문서는 **계획·설계서**이며 구현은 승인 후 진행한다.

---

## 1. life matrix flow — verify 하네스 현황 (조사 결과)

**결론: 하네스를 신규 구축할 필요가 없다 — 캐노니컬 버전에 이미 존재하고 통과한다.**

- **캐노니컬 버전 = `life matrix flow new`** (origin: life-matrix-flow-new, 최신 커밋 2026-06-22). 구버전
  `life matrix flow`(origin: Life-Matrix-Flow, 최신 2026-03-21)는 deprecated.
- `life matrix flow new`는 **Phase 6(2026-06-22)에서 verify 하네스를 이미 구축**했다:
  - `verify/pipelines.repro.test.ts`(Vitest) + `vitest.verify.config.ts` + `npm run verify:pipelines`.
  - 계산 코어를 `utils/pipelineEngine.ts`의 순수 함수 `executePipelineCore`로 추출(동작 변경 없이 headless화).
  - 레퍼런스 `.lifx` 4종(종신+준비금/정기/양로+시나리오/기존 샘플)을 2회 실행해 순보험료·영업보험료·치환수식·
    준비금표·모듈 status까지 **byte-identical** 단언. **실행 결과: 5 tests PASS(1.66s)** ✅(2026-06-23 재확인).
  - 명시적으로 "ML Auto Flow `verify/run-verification.mjs` 개념을 Vitest로 옮긴 것"이라고 README에 기록.

**패러다임:** life matrix new의 모듈은 **생명보험 보험료·준비금 계리 엔진**이다 —
`DefinePolicyInfo · SelectRiskRates · CalculateSurvivors · NxMxCalculator(교환함수) · PremiumComponent ·
NetPremiumCalculator · GrossPremiumCalculator · ReserveCalculator · ScenarioRunner · RateModifier`.
**회귀·분류 등 지도학습 ML 모듈이 없다.** (지도학습 모듈은 deprecated 구버전 `life matrix flow`에만 존재하나
하네스가 없다.)

**Elston 적용 가능성:** Elston의 개선(RandomForest·FeatureEngineer·잔차 진단·CV·순열중요도)은 모두 *지도학습
회귀 파이프라인* 대상이라 life matrix new의 계리 패러다임에 **직접 이식 대상이 아니다(패러다임 불일치).**
- 직접 이식 가능 항목: 없음(부착할 회귀 모듈 자체가 없음).
- *방법론* 전이는 개념적으로만 유효: 향후 life matrix에 **사망률 적합/예측 기능**(Lee–Carter·CBD·APC 등)이
  추가되면, 그때 Elston의 **홀드아웃 검증·잔차 진단** 사고를 적용할 수 있다.

> **권고.** life matrix는 자체 계리 패러다임에 맞춰 **기존 verify 하네스를 확장**(픽스처/시나리오 추가)하는 방향이
> 옳다. Elston ML 모듈을 억지로 이식하지 않는다. (구버전 `life matrix flow`의 RandomForest 템플릿 갭은
> deprecated 버전이라 수정 가치 낮음.)

---

## 2. ★ 추가 작업 감사 — 내보내기 템플릿 누락(RandomForest급 갭, **실증 확인**)

감사 방법: 각 앱 `types.ts`의 지도학습 모듈 ↔ `codeSnippets.ts` 템플릿 키 대조 + `verify/generate.ts`로 **실제
전체코드를 생성해 실증**.

**확인된 버그:** 다음 4개 모델은 캔버스에 배치 가능(TOOLBOX + DEFAULT_MODULES)하고 Python create 함수도
있으나, **`codeSnippets.ts` 전체코드 export 템플릿이 없다.** 결과적으로 "전체 코드 보기"가 그 모델 단계에서
**빈 블록**을 내보내고, 다운스트림 `TrainModel`이 정의되지 않은 `model`에 `model.fit(...)`을 호출 →
**런타임 `NameError`**(외부 재현 불가 = Python 재현성 불변식 위반). RandomForest와 정확히 같은 갭이다.

| 모델 | ML | JMDC | DFA | 증상 |
|---|---|---|---|---|
| LogisticRegression | ❌ 갭 | ❌ 갭 | ❌ 갭 | export 빈 블록 → NameError (실증) |
| SVM | ❌ 갭 | ❌ 갭 | ❌ 갭 | export 빈 블록 → NameError (실증) |
| LDA(LinearDiscriminantAnalysis) | ❌ 갭 | ❌ 갭 | — (모듈 없음) | export 빈 블록 |
| NaiveBayes | ❌ 갭 | ❌ 갭 | ❌ 갭 | export 빈 블록 |

> 참고: `codeSnippets.ts`에 `LogisticTradition` 키가 있으나 `ModuleType.LogisticRegression`과 **키가 일치하지
> 않아** 매핑되지 않는다(LogisticRegression 배치 시 빈 블록 실증). 즉 LogisticRegression도 실질 갭이다.

**영향 범위:** 이 4개 모델을 쓰는 파이프라인의 "전체 코드 보기"·`verify` 픽스처·스코어링 코드 내보내기가
모두 외부 재현 불가. (인앱 실행은 별도 경로라 동작할 수 있으나, 책자가 약속한 "내보낸 코드 재현"이 깨진다.)

**수정안(저위험·검증된 패턴 반복):** RandomForest 수정과 동일하게 —
1. `codeSnippets.ts`에 각 모델 템플릿 추가(DecisionTree/GB/RF 템플릿 미러; `random_state=42` 해당 모델만).
   - LogisticRegression → `sklearn.linear_model.LogisticRegression`(random_state=42)
   - SVM → `sklearn.svm.SVC/SVR`(분류/회귀; SVC 확률 필요시 probability=True는 비결정 주의 → 기본 미사용)
   - LDA → `sklearn.discriminant_analysis.LinearDiscriminantAnalysis`(결정적, 시드 불필요)
   - NaiveBayes → `sklearn.naive_bayes.GaussianNB`(결정적)
2. `data_analysis_modules.py`의 기존 `create_*`와 파라미터·로직 정합 유지.
3. 모델별 `verify/pipelines/*.json` 픽스처 추가(iris/adult 등 기존 데이터) → `verify:pipelines` byte-identical.
4. ML·JMDC byte-identical 동기화, DFA는 3종(LDA 제외) 적용.

**우선순위: ★최우선(RandomForest와 동급 버그).** 난이도 낮음~중. 예상 픽스처 +4(ML/JMDC), +3(DFA).

### 2.1 구현 완료 (2026-06-23)
위 4종 갭을 **수정 완료**했다(원칙 회복 — 모든 지도학습 분석 모듈이 검증 가능한 Python으로 export):
- **ML/JMDC:** LogisticRegression·SVM·LDA·NaiveBayes 템플릿 신설 + LDA를 DEFAULT_MODULES에 추가(누락분) + 픽스처 22~25(iris 분류). `create_*` 함수와 정합, 결정적(random_state는 해당 모델만).
- **DFA:** LogisticRegression·SVM·NaiveBayes 템플릿 신설(LDA는 DFA에 없음) + 파생 분류 데이터 `dfa_claims_class.csv`(claim_amount 중앙값 분할) + 픽스처 05~07.
- **검증:** `verify:pipelines` — **ML 23/23 · JMDC 24/24 · DFA 7/7 PASS**, 3개 앱 build 성공. 공통 코드(템플릿) ML↔JMDC byte-identical.

### 2.2 추가 발견 — DFA 군집(clustering) 패밀리 미완성(원칙 점검 결과)
원칙 종합 감사 중 발견: **DFA의 KMeans/DBSCAN/HierarchicalClustering은 팔레트·DEFAULT_MODULES에 있으나
배치만 가능하고 동작 불가**다 — DFA에는 군집 체인을 완성하는 **`TrainClusteringModel`·`ClusteringData` 모듈
자체가 없고**(ML/JMDC에는 있음), 군집 export 템플릿도 없다. 즉 군집 파이프라인을 구성·검증할 수 없다.
- 성격: 본 작업이 만든 회귀가 아니라 **기존 미완성**(DFA는 회귀/재무 중심이라 군집 패밀리가 미배선).
- **조치(2026-06-23): 팔레트 정리(제거) 완료.** DFA는 회귀·재무 중심이고 군집 포팅은 크고 투기적이라,
  비기능 군집/PCA 4종(KMeans/Hierarchical/DBSCAN/PCA)을 **DFA의 TOOLBOX + DEFAULT_MODULES에서 제거**했다
  (enum·python create_*는 하위호환 위해 유지). 이제 DFA는 *배치 가능한 분석 모듈에 갭 0* — "배치 가능 = 검증
  가능한 Python export"가 성립. build 성공, verify 7/7 유지. 추후 DFA에 군집이 필요해지면 ML/JMDC의 완비된
  군집 패밀리(템플릿 + TrainClusteringModel + ClusteringData + 픽스처 07~10)를 그대로 포팅하면 된다.
- **최종 감사 결과: 3개 앱 모두 배치 가능한 분석 모듈의 export 갭 0**(ML/JMDC의 PrincipalComponentAnalysis는
  `PCA`와 동일 enum 값의 별칭이라 `PCA` 템플릿으로 해석됨 — 갭 아님). 지도학습 핵심 원칙 완전 충족.

---

## 3. 2-7 커스텀 Python 코드 모듈 설계안 (`PythonScript`)

**목적:** Elston의 *Execute Python Script* 대응 — 사용자가 `df → df` 변환/특징생성 코드를 파이프라인에 삽입.
ML·JMDC·DFA(Pyodide 기반)에 적용, life matrix는 비대상(패러다임).

### 3.1 모듈 형태(기존 패턴 재사용)
| 파일 | 변경 |
|---|---|
| `types.ts` | `PythonScript = "PythonScript"` |
| `constants.ts` | TOOLBOX 팔레트 + DEFAULT_MODULES(`parameters:{ code: "<기본 스니펫>" }`, `data_in`→`data_out`) |
| `codeSnippets.ts` | 템플릿: 표준 시그니처로 사용자 `code` 삽입(아래 계약) |
| `utils/generatePipelineCode.ts` | `MODULE_OUTPUT_VAR.PythonScript = 'scripted_data'` |
| `components/PropertiesPanel.tsx` | 코드 에디터(textarea/모노스페이스) + 경고 배너, **AdvancedOnly 게이트** |
| `utils/pyodideRunner.ts` | `runUserScriptPython(rows, code)` — 인앱 실행 |
| `App.tsx` | 가산 else-if 분기(인앱) |
| `verify/pipelines/*.json` | 결정적 스니펫 픽스처 1종 |

### 3.2 실행 계약(표준 시그니처)
```python
# 입력: 'dataframe'(pandas DataFrame) — 직전 모듈 출력
# 사용자 코드는 'dataframe'를 읽어 'scripted_data'(DataFrame)를 만들어야 한다.
# 예) scripted_data = dataframe.assign(log_x=np.log1p(dataframe['x']))
<사용자 code>
# 안전장치: 사용자가 scripted_data를 안 만들면 dataframe을 그대로 통과
if 'scripted_data' not in dir(): scripted_data = dataframe
```

### 3.3 보안 모델 (필수)
- **고급기능 비밀번호 게이트(AdvancedOnly) 의무** — 일반 사용자에게 노출 안 함.
- **기본 비활성·옵트인**, 코드 에디터에 *"임의 코드 실행 — 신뢰할 수 있는 코드만"* 경고 배너.
- **Pyodide 샌드박스**: 브라우저 내 실행, 로컬 파일시스템/네트워크 비노출(`df`만 전달). 서버 실행 없음.
- **import 가이드**: `pandas`·`numpy`·`scikit-learn` 권장. (브라우저 샌드박스라 시스템 접근은 원천 차단되나,
  무한루프·과대 메모리 방지를 위해 60초 타임아웃 적용 — 기존 pyodideRunner 패턴.)
- **재현성 책임 고지**: 내보낸 코드에 사용자 코드가 그대로 포함된다 → 결정성은 사용자 책임. 무작위 사용 시
  `random_state`/시드 고정 가이드를 에디터 도움말에 표기. verify 픽스처는 *결정적 스니펫*만 사용.

### 3.4 리스크 & 완화
| 리스크 | 완화 |
|---|---|
| 임의 코드 실행 | 고급 게이트 + 옵트인 + 샌드박스(브라우저) + 경고 |
| 비결정 코드로 재현성 깨짐 | 시드 고정 가이드 + 결정적 픽스처만 verify + 경고 고지 |
| 무한루프/메모리 | 60초 타임아웃(기존 패턴) |
| 사용자 코드 오류 | try/except로 친절한 오류 메시지 + 통과 폴백 |

### 3.5 단계 구현 계획
1. (P1) types/constants/codeSnippets/MODULE_OUTPUT_VAR + AdvancedOnly 게이트 + verify 픽스처(결정적) → export 재현 검증.
2. (P2) PropertiesPanel 코드 에디터 + 경고 UI.
3. (P3) App.tsx 가산 분기 + pyodideRunner `runUserScriptPython` 인앱 실행.
4. (P4) ML·JMDC byte-identical 동기화, DFA 후순위.

**우선순위: 중(2번 항목 이후).** 보안 검토 동반 권장.

---

## 4. 권고 실행 순서 (추가 작업)

| 순위 | 작업 | 근거 | 난이도 |
|---|---|---|---|
| 1 | **내보내기 템플릿 갭 4종 수정**(LogisticRegression/SVM/LDA/NaiveBayes) | 확인된 버그(NameError)·재현성 불변식 위반·RandomForest 동급 | 낮음~중 |
| 2 | 2-7 커스텀 코드 모듈(`PythonScript`) | Elston 마지막 미구현 항목 | 중(+보안) |
| 3 | life matrix 자체 verify 확장(계리 픽스처 추가) | 패러다임 정합 | 중 |
| — | (비권장) life matrix에 Elston ML 모듈 이식 | 패러다임 불일치 | — |

> **요지.** 이번 점검의 최대 수확은 **#1(4개 export 템플릿 갭)** 이다 — RandomForest와 같은 잠복 버그가
> 4종 더 있었고 실증으로 확인했다. life matrix 하네스는 이미 존재(통과)하므로 별도 구축 불필요.
> 커스텀 코드 모듈(2-7)은 보안 설계를 갖춘 위 설계안대로 진행하면 된다.

---

## 5. 인앱 전용 분석 모듈의 export-Python화 (2026-06-23 완료)

원칙("모든 분석 모듈의 작업이 검증 가능한 Python으로 export") 점검 중, **export가 설정-스텁**이던
분석 모듈 3종을 발견하고 실제 Python으로 교체했다(인앱 전용 → 외부 재현 가능).

| 모듈 | 이전(스텁) | 변경(실제 Python) |
|---|---|---|
| Correlation | `print("...configured...")` | `dataframe[cols].corr('pearson')` 상관행렬 출력 |
| OutlierDetector | `print("...configured...")` | IQR 1.5배 규칙으로 열별 이상치 수·경계 출력 |
| HypothesisTesting | `print("...configured...")` | scipy 검정 디스패치(t/카이제곱/ANOVA/KS/Shapiro/Levene) — 인앱 `performHypothesisTests`와 동일 로직, 결과 출력 |

- 전부 결정적(고정 데이터→고정 통계/p값). 분석 모듈이라 `dataframe`은 변경 없이 통과(MODULE_OUTPUT_VAR=dataframe).
- 픽스처 `26_correlation`·`27_outlier_detector`·`28_hypothesis_testing`(iris) 추가.
- **검증:** ML 26/26 · JMDC 27/27 PASS, build 성공(양쪽). 3종 템플릿 ML↔JMDC byte-identical. (DFA에는 이 3모듈 없음.)
- (잔여) ColumnPlot 등 *시각화* 모듈은 matplotlib 이미지 산출이 본질이라 그림은 앱 표시용으로 두되, 수치 분석은
  Correlation/통계 모듈로 충당된다 — 핵심 *수치 분석*의 검증 가능성은 충족.
