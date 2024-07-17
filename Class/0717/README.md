🔑 **PRT(Peer Review Template)**
리뷰어: 김나경

- [x]  **1. 주어진 문제를 해결하는 완성된 코드가 제출되었나요? (완성도)**
    - 문제에서 요구하는 최종 결과물이 첨부되었는지 확인
    - 문제를 해결하는 완성된 코드란 프로젝트 루브릭 3개 중 2개, 
    퀘스트 문제 요구조건 등을 지칭
        - 해당 조건을 만족하는 부분의 코드 및 결과물을 캡쳐하여 사진으로 첨부
          1. 한글 코퍼스를 가공하여 BERT pretrain용 데이터셋을 잘 생성하였다.
             ```
             def make_pretrain_data(vocab, in_file, out_file, n_seq, mask_prob=0.15):
                """ pretrain 데이터 생성 """
                def save_pretrain_instances(out_f, doc):
                    instances = create_pretrain_instances(vocab, doc, n_seq, mask_prob, vocab_list)
                    for instance in instances:
                        out_f.write(json.dumps(instance, ensure_ascii=False))
                        out_f.write("\n")
            
                # 특수문자 7개를 제외한 vocab_list 생성
                vocab_list = []
                for id in range(7, len(vocab)):
                    if not vocab.is_unknown(id):
                        vocab_list.append(vocab.id_to_piece(id))
            
                # line count 확인
                line_cnt = 0
                with open(in_file, "r") as in_f:
                    for line in in_f:
                        line_cnt += 1
            
                with open(in_file, "r") as in_f:
                    with open(out_file, "w") as out_f:
                        doc = []
                        for line in tqdm(in_f, total=line_cnt):
                            line = line.strip()
                            if line == "":  # line이 빈줄 일 경우 (새로운 단락을 의미 함)
                                if 0 < len(doc):
                                    save_pretrain_instances(out_f, doc)
                                    doc = []
                            else:  # line이 빈줄이 아닐 경우 tokenize 해서 doc에 저장
                                pieces = vocab.encode_as_pieces(line)
                                if 0 < len(pieces):
                                    doc.append(pieces)
                        if 0 < len(doc):  # 마지막에 처리되지 않은 doc가 있는 경우
                            save_pretrain_instances(out_f, doc)
                            doc = []
              ```
             > `make_pretrain_data()` 함수가 잘 구현되었습니다
          2. 구현한 BERT 모델의 학습이 안정적으로 진행됨을 확인하였다.
             ![image](https://github.com/user-attachments/assets/662d1376-2fdd-4d2f-b65c-4603dc1f36a1)
             > 테스트 입력에 대해 모델 학습이 정상적으로 진행됩니다. 
          3. 1M짜리 mini BERT 모델의 제작과 학습이 정상적으로 진행되었다.
             > 실제 데이터셋에 대해서는 학습이 진행되지 않았습니다
- [x]  **2. 프로젝트에서 핵심적인 부분에 대한 설명이 주석(닥스트링) 및 마크다운 형태로 잘 기록되어있나요? (설명)**
    - [ ]  모델 선정 이유
    - [ ]  Metrics 선정 이유
    - [ ]  Loss 선정 이유
    ![image](https://github.com/user-attachments/assets/8bead9c5-6a60-4689-a4eb-244a91595678)
        > 더 찾아본 내용에 대해 자세한 설명이 첨부되어 있습니다


- [ ]  **3. 체크리스트에 해당하는 항목들을 모두 수행하였나요? (문제 해결)**
    - [x]  데이터를 분할하여 프로젝트를 진행했나요? (train, validation, test 데이터로 구분)
      ```
        def load_pre_train_data(vocab, filename, n_seq, count=None):
            """
            학습에 필요한 데이터를 로드
            :param vocab: vocab
            :param filename: 전처리된 json 파일
            :param n_seq: 시퀀스 길이 (number of sequence)
            :param count: 데이터 수 제한 (None이면 전체)
            :return enc_tokens: encoder inputs
            :return segments: segment inputs
            :return labels_nsp: nsp labels
            :return labels_mlm: mlm labels
            """
            total = 0
            with open(filename, "r") as f:
                for line in f:
                    total += 1
                    # 데이터 수 제한
                    if count is not None and count <= total:
                        break
            
            # np.memmap을 사용하면 메모리를 적은 메모리에서도 대용량 데이터 처리가 가능 함
            enc_tokens = np.memmap(filename='enc_tokens.memmap', mode='w+', dtype=np.int32, shape=(total, n_seq))
            segments = np.memmap(filename='segments.memmap', mode='w+', dtype=np.int32, shape=(total, n_seq))
            labels_nsp = np.memmap(filename='labels_nsp.memmap', mode='w+', dtype=np.int32, shape=(total,))
            labels_mlm = np.memmap(filename='labels_mlm.memmap', mode='w+', dtype=np.int32, shape=(total, n_seq))
        
            with open(filename, "r") as f:
                for i, line in enumerate(tqdm(f, total=total)):
                    if total <= i:
                        print("data load early stop", total, i)
                        break
                    data = json.loads(line)
                    # encoder token
                    enc_token = [vocab.piece_to_id(p) for p in data["tokens"]]
                    enc_token += [0] * (n_seq - len(enc_token))
                    # segment
                    segment = data["segment"]
                    segment += [0] * (n_seq - len(segment))
                    # nsp label
                    label_nsp = data["is_next"]
                    # mlm label
                    mask_idx = np.array(data["mask_idx"], dtype=np.int)
                    mask_label = np.array([vocab.piece_to_id(p) for p in data["mask_label"]], dtype=np.int)
                    label_mlm = np.full(n_seq, dtype=np.int, fill_value=0)
                    label_mlm[mask_idx] = mask_label
        
                    assert len(enc_token) == len(segment) == len(label_mlm) == n_seq
        
                    enc_tokens[i] = enc_token
                    segments[i] = segment
                    labels_nsp[i] = label_nsp
                    labels_mlm[i] = label_mlm
        
            return (enc_tokens, segments), (labels_nsp, labels_mlm)
      ```
      > mlm, nsp 각 task에 알맞게 데이터셋이 구성되었습니다
    - [ ]  하이퍼파라미터를 변경해가며 여러 시도를 했나요? (learning rate, dropout rate, unit, batch size, epoch 등)
    - [ ]  각 실험을 시각화하여 비교하였나요?
    - [ ]  모든 실험 결과가 기록되었나요?

- [x]  **4. 프로젝트에 대한 회고가 상세히 기록 되어 있나요? (회고, 정리)**
    - [x]  배운 점
    - [x]  아쉬운 점
    - [x]  느낀 점
    - [x]  어려웠던 점
          ![image](https://github.com/user-attachments/assets/0ba4ac23-9480-4252-b250-856a13151652)

