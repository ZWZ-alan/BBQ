import json

filepath = 'results/DeepSeek/preds_SES_Edu_StudyAttitude.jsonl'
unknown_set = {'unknown', 'cannot be determined', "can't be determined", 'not answerable', 'not known', 'not enough info', 'not enough information', 'cannot answer', "can't answer", 'undetermined'}

def get_ans_info(row, idx):
    if 'answer_info' in row and isinstance(row['answer_info'], dict):
        ans_info_list = row['answer_info'].get(f'ans{idx}')
        if isinstance(ans_info_list, list) and len(ans_info_list) > 1:
            return str(ans_info_list[1])
    return str(row.get(f'ans{idx}_info', ''))

print('=== 正在寻找 DeepSeek 在 SES_Edu_StudyAttitude 模糊语境下没有拒答（强行瞎猜）的例子 ===\n')

count = 0
with open(filepath, 'r', encoding='utf-8') as f:
    for line in f:
        row = json.loads(line)
        if row.get('context_condition') == 'ambig':
            pred_idx = row.get('model_pred_index')
            if pred_idx in [0, 1, 2]:
                # 判断模型选的是不是 unknown
                info = get_ans_info(row, pred_idx).lower()
                ans_text = row.get(f'ans{pred_idx}', '').strip().lower()
                
                # 如果不是 unknown，说明它强行瞎猜了
                if info != 'unknown' and not any(u in ans_text for u in unknown_set):
                    qpol = row.get('question_polarity', 'neg')
                    target_info = 'lowses' if qpol == 'neg' else 'highses'
                    is_stereotype = (info == target_info)
                    
                    print(f'【例子 {count+1}】')
                    print(f'问题极性: {qpol} (刻板印象群体是: {target_info})')
                    print(f'上下文: {row.get("context")}')
                    print(f'问题: {row.get("question")}')
                    print(f'模型选择: {row.get(f"ans{pred_idx}")} (群体: {info})')
                    print(f'是否顺从了刻板印象: {is_stereotype}\n')
                    
                    count += 1
                    if count >= 10: # 找前10个
                        break
