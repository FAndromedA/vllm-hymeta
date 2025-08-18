from transformers import AutoTokenizer, AutoModelForCausalLM

TOKENIZER_PATH='/root/zhuangjh/modelbase/V1_7B_sft_s3_reasoning'
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, padding_side='left', trust_remote_code=True)
model  = AutoModelForCausalLM.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
device = 'cuda'
model = model.to(device)
left_padded_input = tokenizer(["Hello world", "Hi"],  return_tensors="pt", padding=True)
left_padded_input = left_padded_input.to(device)
print(left_padded_input)
outputs = model.generate(input_ids = left_padded_input.input_ids,
                         attention_mask = left_padded_input.attention_mask,
                         max_new_tokens=5, use_cache=True, 
                         do_sample= False,
                         past_key_values=None,
                         temperature=0)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))