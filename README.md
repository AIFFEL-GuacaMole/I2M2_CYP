# I2M2_CYP

## 프로젝트 개요

**I2M2_CYP** 프로젝트는 TDC의 `CYP2C19_Veith` 데이터를 활용해 **CYP 이진 분류** 문제를 해결하기 위한 다양한 학습 구조를 시도합니다.  
- **Unimodal 모델**: ChemBERT(`chembert_binary_classifier`), 1D CNN+GRU(`cnn_gru_binary_classifier`)  
- **Intra Modality**: 같은 모달 내 여러 모델 앙상블/학습  
- **Inter Modality**: 서로 다른 모달(예: BERT, CNN) 결합 (Fusion)  
- **Inter + Intra Modality**: 두 기법을 혼합한 I2M2 방식


---

## 폴더 구조

```
I2M2_CYP/
  ├─ data/
  │   ├─ cyp2c19_train.csv  
  │   ├─ cyp2c19_valid.csv  
  │   └─ cyp2c19_test.csv  
  ├─ datasets/  
  │   ├─ __init__.py  
  │   └─ data_loader.py              
  ├─ unimodal/  
  │   ├─ __init__.py  
  │   ├─ chembert_binary_classifier.py    
  │   └─ cnn_gru_binary_classifier.py     
  ├─ training_structures/  
  │   ├─ __init__.py  
  │   ├─ unimodal.py                 
  │   ├─ intra_modality.py           
  │   ├─ inter_modality.py           
  │   └─ inter_and_intra_modality.py  
  ├─ common_fusions/  
  │   ├─ __init__.py  
  │   └─ fusions.py                  
  ├─ main.py                         
  └─ ckpts/                          
```

---

## 실행 예시

1. **Unimodal (CNN+GRU) 학습**  
   ```bash
   python main.py --model_type unimodal --unimodal_arch cnn_gru        --train --epochs 5 --batch_size 16
   ```
   - 모델 학습이 완료되면 `./ckpts/unimodal_cnn_gru.pt`에 파라미터가 저장됩니다.

2. **Unimodal (ChemBERT) 학습**  
   ```bash
   python main.py --model_type unimodal --unimodal_arch chembert        --train --epochs 5 --batch_size 16
   ```
   - 학습 완료 후, `./ckpts/unimodal_chembert.pt`에 파라미터 저장

3. **Intra Modality**  
   ```bash
   python main.py --model_type intra --train --epochs 5 --batch_size 16
   ```
   - 같은 모달(또는 ChemBERT + CNN)을 앙상블하여 학습

4. **Inter Modality**  
   ```bash
   python main.py --model_type inter --train --epochs 5 --batch_size 16
   ```
   - 서로 다른 모달(encoders) + fusion 방식으로 학습

5. **Inter + Intra Modality (I2M2)**  
   ```bash
   python main.py --model_type inter_intra --train --epochs 5 --batch_size 16
   ```
   - Inter 모달 모델과 unimodal(또는 intra) 모델을 함께 활용

실행 시, 학습이 종료되면 검증(Val) 지표(Accuracy 등)가 표시되며, 최적 파라미터는 `./ckpts/` 폴더에 저장됩니다.  
학습이 끝난 모델을 테스트하려면, `--test` 플래그를 추가하여 체크포인트 로드 후 성능을 확인할 수 있습니다.