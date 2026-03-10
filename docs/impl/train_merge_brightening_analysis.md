# train_merge 결과가 밝아지는 경향: 원인 분석

`train_merge.py` 실행 결과가 전체적으로 밝아지는 현상에 대해, **merge 알고리즘**, **cost 계산**, **파이프라인 설정** 측면에서 가능한 원인을 정리한다.

---

## 1. 파이프라인/렌더 설정 (가장 유력)

### 1.1 Exposure 미적용

- **train_merge**: `render()` 호출 시 `use_trained_exp`를 넘기지 않음 → 기본값 False.  
  따라서 **per-camera exposure가 렌더에 적용되지 않음** (또는 항상 identity로 취급).
- **train_and_prune**: `use_trained_exp=dataset.train_test_exp` 등으로 학습 시 exposure를 학습·적용하고, 테스트 시 `rendered_image = matmul(rendered_image, exposure[:3,:3]) + exposure[:3,3]` 로 적용.
- **init PLY**: `load_ply(..., use_train_test_exp=False)` 이므로 **원본 실험의 exposure를 로드하지 않음**.  
  원본이 exposure를 써서 어둡게 맞춰둔 SH라면, train_merge에서는 identity로 렌더되므로 **같은 SH가 더 밝게 보임**.
- **결론**: 비교 대상이 “exposure 적용된 GHAP 본학습 결과”라면, train_merge 쪽이 **exposure를 안 써서** 전체적으로 밝게 보일 수 있다.

### 1.2 배경

- train_merge는 `bg_color = [0,0,0]` 고정.  
  본학습이 `white_background`이면 배경만 다르고, 픽셀 값 자체의 체계적 밝기 증가 원인은 아님.  
  (시각 비교 시 인지적 대비는 있을 수 있음.)

---

## 2. Merge 알고리즘 (merge_3dgs.py)

### 2.1 Opacity: “절반 질량” 보존

- **공식**:  
  `mass = (a_u*vol_u + a_v*vol_v) * 0.5`  
  `a_new = mass / vol_new`  
  → **α_new = (α_u·vol_u + α_v·vol_v) / (2·vol_new)**.
- 즉, 두 가우시안의 “질량”(α×vol) 합의 **절반**만 merged Gaussian에 부여한다.  
  나머지 절반은 **의도적으로 버림** (2D alpha compositing처럼 α가 단조증가하는 것을 막기 위함).
- **영향**: merge 직후에는 **같은 영역이 더 어두워지는 쪽**이다 (opacity 감소).  
  따라서 merge 자체는 “밝게 만드는” 쪽이 아니라 “어둡게 만드는” 쪽이다.

### 2.2 SH/색상

- **f_dc, f_rest**: `wu = a_u/(a_u+a_v)`, `wv = a_v/(a_u+a_v)` 로 **opacity 가중 평균**.  
  산술 평균이 아니라 opacity 비율로 섞이므로, 공식 자체에서 체계적으로 밝기를 올리는 항은 없다.

### 2.3 Merge가 밝기를 올리는 간접 경로

- Merge가 반복되면 **opacity가 계속 줄어드는 쪽**으로 작동한다.
- 학습은 **동일 GT**에 맞추려 하므로, **손실이 “너무 어두움”을 penalize**한다.
- 그 결과 **opacity·SH가 보상적으로 커지는 방향**으로 업데이트될 수 있고, merge 주기마다 “merge로 어두워짐 → 학습으로 다시 밝게”를 반복하다 **보상이 과해져** 전체적으로 밝아지는 drift가 생길 수 있다.

---

## 3. Cost 계산 (edge_cost_3dgs.py)

### 3.1 Cost의 “질량” vs Merge의 “질량”

- **Cost**:  
  `V_q = V_u + V_v` (완전 질량 보존),  
  `a_Q` 는 1을 넘을 수 있음 (평가용).
- **Merge**:  
  실제 적용은 `α_new = (α_u·vol_u + α_v·vol_v) / (2·vol_new)` 로 **절반 질량**만 부여.
- **불일치**:  
  Cost는 “두 개를 하나로 합쳤을 때 **전부** 보존하면 L2가 이만큼”이라고 보고, 그 L2가 threshold 이하인 edge를 고른다.  
  그런데 실제 merge는 **절반만** 보존하므로, merge 직후 예측보다 **더 어둡다**.  
  → “Cost로는 괜찮다고 골랐는데, 적용하면 생각보다 어둡다”가 반복되고, 학습이 계속 밝기를 올리는 쪽으로 보상할 여지가 생긴다.

- Cost는 **V_q = V_u + V_v** (전체 질량) 유지. 겹친 두 P를 하나의 Q로 덮을 수 있어야 L2가 타당함; 반쪽 질량이면 불가해 되돌림. Merge만 반쪽 opacity 사용 → drift 가능성은 exposure/threshold로 대응.

### 3.2 Cost 자체의 색/밝기 편향

- Cost는 **geometry + SH(색)** 의 L2² (volume 정규화, bloat/size-mismatch penalty 등).  
  “밝은 걸 먼저 merge” 또는 “어두운 걸 먼저 merge”하는 **명시적 선택**은 없다.
- 다만 **volume 정규화**로 인해 큰 가우시안 쌍이 상대적으로 merge되기 쉬운 구조일 수 있고, 그에 따른 간접적 분포 변화는 가능하다.  
  하지만 “전체가 밝아진다”를 직접 설명하는 인과는 weak하다.

---

## 4. Pruning

- `iteration > 500` 부터 `sigmoid(opacity) < 0.005` 인 점을 주기적으로 제거.
- **저 opacity** 포인트는 보통 **어둡거나 보조적인 영역**일 가능성이 크다.  
  이들을 제거하면 **상대적으로 더 두드러진(밝은) 가우시안 비중이 커져**, 전체 톤이 살짝 밝아지는 효과는 있을 수 있다.  
  주된 원인이라기보다 **보조 요인** 정도로 보는 것이 타당하다.

---

## 5. 요약 및 대응

| 원인 | 방향 | 대응 예시 |
|------|------|-----------|
| **Exposure 미적용** | 렌더가 원본 대비 밝게 나옴 | init PLY 출처 실험과 동일하게 `use_trained_exp` 적용, 필요 시 `load_ply(..., use_train_test_exp=True)` 및 exposure 로드 |
| **Cost 질량 (V_q = V_u+V_v 유지, drift는 exposure/threshold로)** | merge 시 “의도보다 어두움” → 학습이 밝게 보상 | merge 시 질량 보존을 cost와 맞추거나 (예: 분모 2 제거 검토), 또는 cost에서도 동일한 “절반 질량” 가정 사용 |
| **Merge 반복 + 학습 보상** | merge로 어두워짐 → 학습으로 과보상 → drift | merge 직후 일시적으로 opacity/색 보정, 또는 merge 빈도/강도 완화 |
| **저-opacity pruning** | 어두운 점 제거 → 평균 밝기 소폭 상승 | pruning threshold 상향(덜 aggressive) 또는 merge 구간과 분리해 실험 |

**실무적으로는**  
1) **exposure/배경을 train_and_prune과 동일하게 맞춘 뒤** (문서 `train_merge_vs_train_and_prune.md` 참고)  
2) cost는 전체 질량(V_q = V_u+V_v) 유지, merge만 체적 밀도 opacity 사용. 여전히 밝으면 exposure/배경·pruning·merge 빈도 등을 검토하는 순서를 권장한다.
