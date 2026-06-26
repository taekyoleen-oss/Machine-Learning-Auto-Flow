---
name: model-analysis-report
description: ML Auto Flow / JMDC에서 완성된 모델 파이프라인의 "개발 과정 문서"를 자기완결 HTML 보고서로 생성하는 방법론. 인앱 신규 모듈 "모델 분석보고서"(ModelAnalysisReport)의 사양·HTML 템플릿·AI 프롬프트·결정적 폴백·웹 리서치 폴백·고급사용자 게이트를 정의한다. "모델 분석보고서", "개발과정 문서화", "모델 리포트 HTML", "분석보고서 모듈" 작업이면 반드시 이 스킬을 참조하라.
---

# 모델 분석보고서 (Model Analysis Report)

모델 파이프라인을 **개발한 뒤**, 그 데이터·과정·결과·해석을 **자기완결 HTML 문서**로 남기는 기능. 두 가지 산출물을 다룬다.

1. **인앱 모듈 "모델 분석보고서"**(`ModelAnalysisReport`) — 파이프라인 **맨 끝**에 두는 모듈. 업스트림 파이프라인을 자동 수집 + 사용자 추가정보(PDF/텍스트) + (입력이 없으면) 웹 리서치로 보강하여 **AI로 HTML 보고서를 생성**하고 모듈 결과로 저장한다. 일반 사용자는 결과를 열람만, **실행은 고급 사용자만**.
2. **에이전트/스킬 경로** — Claude(나)가 직접 같은 방법론으로 HTML 보고서를 작성(웹 도구 사용 가능). 첫 레퍼런스 산출물: `docs/reports/adult_census_income_decisiontree_분석보고서.html`.

기준 예시(반드시 먼저 열어볼 것): **`docs/reports/adult_census_income_decisiontree_분석보고서.html`** — 스타일·섹션 구성의 정답지.

---

## 1. 불변식 / 위치

- 이 모듈은 **데이터 분석 모듈이 아니라 문서화(메타) 모듈**이다. AI 산출물은 비결정적이므로 **`codeSnippets.ts` export·`verify:pipelines` 대상이 아니다**(Python 재현성 불변식과 무관, 위반 아님). 단, 보고서가 인용하는 수치는 업스트림의 **결정적 Python 결과**에서 가져온다.
- 인앱 분류상 AI/문서 기능(예: PythonScript·AI 파이프라인)과 동급의 **앱 전용·가산적** 기능이다.
- 출력 HTML은 **자기완결**(인라인 CSS, 외부 의존 없음)이어야 모듈 결과로 저장·열람·다운로드가 가능하다.

---

## 2. 입력 우선순위 (★웹 리서치 폴백 포함)

보고서 본문 자료는 다음 우선순위로 확보한다. 상위가 있으면 우선, 부족분은 하위로 보강한다.

1. **사용자 추가정보** — 모듈에 업로드한 PDF(브라우저 pdf.js로 텍스트 추출) 또는 붙여넣은 텍스트/마크다운(데이터 속성·구조·도메인 설명). 최우선 근거.
2. **파이프라인 자동 수집 메타데이터** — 항상 수집(아래 3장). 수치·하이퍼파라미터·지표의 1차 출처.
3. **웹 리서치 폴백** — 1번 입력이 없거나 빈약하면 인터넷 조사로 데이터셋·도메인 배경을 확보한다. *대부분의 공개 데이터셋(UCI 등)·기법은 웹에 잘 문서화돼 있다.* 
   - **에이전트/스킬 경로:** `WebSearch`/`WebFetch`로 데이터셋 출처·컬럼 정의·도메인 배경·기법 설명을 조사하고 **출처 URL을 보고서에 명시**(각주/참고문헌). 수치는 파이프라인 실측을 우선하고, 웹 자료는 배경 설명에만 사용한다(웹 수치로 실측을 덮어쓰지 말 것).
   - **인앱 모듈 경로:** 브라우저에서 임의 웹 검색은 불가하므로 ① Gemini의 일반 지식으로 데이터셋/도메인 배경을 서술하도록 프롬프트하고(데이터셋 이름·도메인을 메타데이터로 전달), ② (선택) 향후 검색 그라운딩/`/api/proxy-csv`류 서버 프록시로 확장한다. 키가 없으면 결정적 템플릿이 메타데이터만으로 채운다.
   - 웹/AI 일반지식에서 온 서술은 **"(웹 출처: URL)" 또는 "(일반 지식 기반)"**으로 표기해 실측과 구분한다.

---

## 3. 파이프라인 메타데이터 자동 수집 (인앱 모듈)

모듈 실행 시 `modules` + `connections`로 `report_in`에 연결된 최종 모듈에서 **역방향으로 그래프를 거슬러** 다음을 수집한다(예시 매핑은 `Adult` 보고서 참조).

| 항목 | 출처 모듈 | 수집 내용 |
|------|-----------|-----------|
| 데이터셋 | LoadData/XolLoading | `source`(파일명), 행/열 수, 컬럼명·타입, (가능시) 클래스 분포 |
| 분할 | SplitData | `train_size`·`random_state`·`shuffle`·`stratify`, train/test 규모 |
| 모델 정의 | DecisionTree/RandomForest/Logistic/… | 모델 종류 + 하이퍼파라미터 전부 |
| 학습 | TrainModel | `feature_columns`·`label_column`·`model_purpose` |
| 채점 | ScoreModel | 예측/확률 컬럼 존재 여부 |
| 평가 | EvaluateModel | `model_type`, 지표(정확도/정밀도/재현율/F1/AUC/AP 또는 회귀 MSE/R²…), 혼동행렬, 임계값 스윕 |
| 군집(해당시) | KMeans/DBSCAN/… + AssignClusters | k·실루엣·관성·클러스터 분포 |

수집 결과는 `ReportContext` 객체(JSON)로 직렬화해 AI 프롬프트/결정적 템플릿에 전달한다.

---

## 4. 보고서 표준 구조 (모델 유형별 적응)

`Adult` 예시의 8개 섹션을 기본 골격으로 하되, 모델 유형에 맞게 6장(결과·해석)을 적응한다.

1. **요약** — 과제 유형·데이터·핵심 지표 KPI·핵심 메시지(불균형/주의점).
2. **데이터셋 개요** — 이름·출처·규모·과제·타깃 + 원본 표본 표.
3. **변수(컬럼) 사전** — 컬럼별 유형·의미·예시. **사용된 특성 vs 미사용 특성** 명시.
4. **타깃/클래스 분포** (분류) 또는 **타깃 분포·기술통계** (회귀) 또는 **클러스터 수·분포** (군집).
5. **모델 개발 과정** — 파이프라인 다이어그램 + 모듈 단계별 역할·파라미터.
6. **분석 결과와 해석** — *분류:* 혼동행렬·임계값 스윕·ROC/AUC·PR. *회귀:* MSE/RMSE/MAE/R²·잔차·실제vs예측. *군집:* 실루엣·관성·클러스터 프로파일.
7. **재현성** — 시드 고정·verify·내보낸 Python.
8. **결론 및 한계** — 핵심 결론 + 개선 방향(미사용 특성·불균형 보정·대안 모델).

수치를 **지어내지 말 것**. 메타데이터/실측에 없는 값은 비우거나 "(자료 없음)"으로 둔다.

---

## 5. HTML 템플릿 & 스타일

- 정답 스타일: `docs/reports/adult_census_income_decisiontree_분석보고서.html`의 `<style>` 블록(인라인 CSS, CJK 폰트 스택 포함). 결정적 폴백·AI 생성 모두 **이 스타일을 재사용**한다.
- 골격 템플릿(플레이스홀더): `templates/report_template.html`.
- 필수 요소: 표지(badge "모델 분석보고서" + 제목·부제·메타), `.callout`(목적/핵심/주의), `.kpi-grid`(핵심 지표), 표(데이터·변수사전·혼동행렬·임계값), `pre`(파이프라인 다이어그램), `footer`(출처).
- **자기완결**: 외부 CSS/JS/폰트 금지(시스템 폰트 스택만). 모듈 출력으로 단일 문자열 저장 → sandbox `<iframe srcdoc>`로 렌더.

---

## 6. AI 프롬프트 설계 (Gemini, `lib/aiHelpers.ts`)

새 헬퍼 `generateModelReportHtml(ctx: ReportContext, extraInfo: string, opts)` 권장. 프롬프트 원칙:

- **System:** "너는 ML 모델 문서 작성가다. 주어진 파이프라인 메타데이터와 추가정보만으로, 한국어 **자기완결 HTML** 모델 분석보고서를 작성한다. 제공된 CSS 스타일과 섹션 구조를 그대로 사용한다."
- **User:** `ReportContext`(JSON) + 추가정보(PDF/텍스트) + (선택)웹 리서치 요약 + 템플릿 스타일을 전달.
- **강제 규칙:** ① 메타데이터에 있는 수치만 사용(창작 금지). ② 데이터셋/도메인 배경 서술은 일반지식 허용하되 "(일반 지식 기반)"/"(웹 출처)" 표기. ③ 출력은 `<!DOCTYPE html>`로 시작하는 완전한 HTML 1개. ④ 표·callout·KPI 적극 활용. ⑤ 미사용 특성·불균형 등 한계를 정직히 기술.
- 출력 검증: `<html>`·`</html>` 포함, `<script>` 없음(보안), 길이 상한. 실패 시 결정적 폴백으로 강등.

---

## 7. 결정적 템플릿 폴백 (API 키 없을 때)

`buildModelReportHtmlFallback(ctx, extraInfo)` — AI 없이 `ReportContext`만으로 표준 구조 HTML을 조립한다. 메타데이터를 표/KPI/문단에 직접 바인딩(창작 0). 키가 없거나 AI 실패 시 항상 이 결과를 반환해 **일반 사용자도 결과를 열람**할 수 있게 한다(빈 결과 금지).

---

## 8. 인앱 모듈 사양 (Phase 2 구현 청사진)

| 영역 | 내용 |
|------|------|
| `types.ts` | `ModuleType.ModelAnalysisReport = "ModelAnalysisReport"`; `interface ModelReportOutput { type:"ModelReportOutput"; html:string; generatedAt:string; source:"ai"|"fallback"; meta:ReportContext }` |
| `constants.ts` | TOOLBOX_MODULES + DEFAULT_MODULES 항목. 한글 표시명 **"모델 분석보고서"**. 포트: 입력 `report_in`(model/eval/cluster 출력 수용), 출력 없음(말단). 파라미터: `extra_info`(textarea), `extra_pdf`(파일, pdf.js 텍스트추출), `title`, `use_web_research`(bool) |
| `App.tsx` | 실행 분기(가산 else-if): 메타데이터 수집 → 추가정보/웹 → `generateModelReportHtml` 또는 폴백 → `outputData={type:"ModelReportOutput",...}`. **실행 트리거는 `AdvancedOnly` 게이트**(일반 사용자는 실행 버튼 비표시/비활성). |
| `components/ModelReportPreviewModal.tsx` | 결과 HTML을 sandbox `<iframe srcdoc>`로 렌더 + **다운로드(.html)**·인쇄 버튼. **열람은 게이트 없음**(일반 사용자 가능). |
| `lib/aiHelpers.ts` | `generateModelReportHtml` + `buildModelReportHtmlFallback` |
| 게이트 | `contexts/AdvancedFeatureContext`의 `AdvancedOnly`/`useAdvancedFeature`로 **실행만** 제한 |
| 양쪽 앱 | ML↔JMDC 공통 가산, byte-identical(모달·헬퍼·타입), JMDC 전용 보존. DFA/life matrix는 후속 |

말단 모듈이므로 export/verify 무관(메타 모듈). 모듈 추가 패턴은 기존 신규모듈 이력(CLAUDE.md) 참조.

---

## 9. 작업 절차 (에이전트/스킬 경로로 보고서 1건 생성)

1. 대상 파이프라인/샘플 파악(모듈·파라미터·실제 실행 수치). 필요시 앱을 실제 실행해 지표 확보.
2. 추가정보(PDF/텍스트) 있으면 우선 반영. 없으면 **웹 리서치**(`WebSearch`/`WebFetch`)로 데이터셋·도메인 배경 확보(출처 명시).
3. 4장 구조로 내용 구성, 5장 스타일의 자기완결 HTML로 작성(`Adult` 예시 미러).
4. 수치는 실측만, 배경 서술은 출처 표기. `docs/reports/<name>.html`로 저장.
5. (선택) `make-pdf`로 PDF 동시 산출.

---

## 체크리스트
- [ ] HTML이 `<!DOCTYPE html>` 자기완결(외부 의존 0, `<script>` 0)인가?
- [ ] 모든 수치가 파이프라인 실측/메타데이터와 일치하는가(창작 0)?
- [ ] 사용/미사용 특성, 불균형 등 한계를 정직히 기술했는가?
- [ ] 입력이 없을 때 웹 리서치(또는 AI 일반지식)로 배경을 채우고 **출처를 표기**했는가?
- [ ] 실행은 고급 사용자만, 열람은 일반 사용자도 가능한가?
- [ ] 키 없을 때 결정적 폴백으로 결과가 비지 않는가?
