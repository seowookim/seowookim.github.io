🔑 **PRT(Peer Review Template)**  
작성자: 김나경 


- [x]  **1. 주어진 문제를 해결하는 완성된 코드가 제출되었나요? (완성도)**
    - 문제에서 요구하는 최종 결과물이 첨부되었는지 확인
    - 문제를 해결하는 완성된 코드란 프로젝트 루브릭 3개 중 2개, 
    퀘스트 문제 요구조건 등을 지칭
        - 해당 조건을 만족하는 부분의 코드 및 결과물을 캡쳐하여 사진으로 첨부
          1. 모델과 데이터를 정상적으로 불러오고, 작동하는 것을 확인하였다.
            ![image](https://github.com/user-attachments/assets/b46d3ddd-48c7-487f-b528-72167cdd00b5)
            ![image](https://github.com/user-attachments/assets/679bc0ab-11bb-4ff9-b5ff-c78bf2e0f527)
            > 알맞은 모델과 데이터를 잘 불러옴

          2. Preprocessing을 개선하고, fine-tuning을 통해 모델의 성능을 개선시켰다.	
             ![image](https://github.com/user-attachments/assets/428bb03d-93f6-44a3-b16d-85a90f81615e)
             > 정확도 90%은 달성하지 못

          4. 모델 학습에 Bucketing을 성공적으로 적용하고, 그 결과를 비교분석하였다.
             ```
             from transformers import DataCollatorWithPadding

             data_collator = DataCollatorWithPadding(tokenizer=huggingface_tokenizer, padding=True)
             ```
             ```
             training_arguments = TrainingArguments(output_dir, evaluation_strategy = "epoch", learning_rate = 2e-5, group_by_length=True,
                                      per_device_train_batch_size = 8, per_device_eval_batch_size = 8, num_train_epochs = 3, 
                                      weight_decay = 0.01)
             ```
             ```
             trainer = Trainer(model = huggingface_model, args = training_arguments, train_dataset = train_dataset_2, eval_dataset = test_dataset_2, data_collator = data_collator, compute_metrics = compute_metrics) 
             trainer.train()
             ```
             > `DataCollatorWithPadding`와 `group_by_length` 옵션이 잘 적용됨
- [x]  **2. 프로젝트에서 핵심적인 부분에 대한 설명이 주석(닥스트링) 및 마크다운 형태로 잘 기록되어있나요? (설명)**
    - [ ]  모델 선정 이유
    - [ ]  Metrics 선정 이유
    - [해당 없음]  Loss 선정 이유
      ![image](https://github.com/user-attachments/assets/60e4f919-7ea7-4bed-9840-4092cccd599b)
      > 단계 별로 마크다운이 잘 적혀 있음


- [x]  **3. 체크리스트에 해당하는 항목들을 모두 수행하였나요? (문제 해결)**
    - [x]  데이터를 분할하여 프로젝트를 진행했나요? (train, validation, test 데이터로 구분)
          ![image](https://github.com/user-attachments/assets/966945f1-e35c-45d9-9b1c-6b535c5c436d)
          > 데이터를 분할하였으며 학습 시간 조절을 위해 일부만 사용
    - [x]  하이퍼파라미터를 변경해가며 여러 시도를 했나요? (learning rate, dropout rate, unit, batch size, epoch 등)
          ![image](https://github.com/user-attachments/assets/671a2efa-7236-48f5-8dd1-2548dc1411c8)
          > 토큰화에서 max_length를 변경해가며 시도함
    - [ ]  각 실험을 시각화하여 비교하였나요?
    - [x]  모든 실험 결과가 기록되었나요?
          ![image](https://github.com/user-attachments/assets/6051f481-bc48-4172-bc1c-3a86236175fe)
         > 실험 별, epoch 별로 loss와 accuracy, F1 score가 잘 기록됨


- [x]  **4. 프로젝트에 대한 회고가 상세히 기록 되어 있나요? (회고, 정리)**
    - [x]  배운 점
    - [x]  아쉬운 점
    - [x]  느낀 점
    - [ ]  어려웠던 점
    ![image](https://github.com/user-attachments/assets/bd2323ee-3a31-4248-947d-d13efc119f0f)
