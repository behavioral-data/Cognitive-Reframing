import os
import pandas as pd
import numpy as np
from random import shuffle
import time
import argparse
from random import shuffle
from tqdm import tqdm

import torch
from sentence_transformers import SentenceTransformer, util

'''
Arguments
'''
parser = argparse.ArgumentParser()
parser.add_argument('--training_path', type=str, default='data/reframing_dataset.csv', help='Path to the training data')
parser.add_argument('--test_path', type=str, default='data/sample_test.csv', help='Path to the test data')
parser.add_argument('--output_path', type=str, default='data/sample_test_output.csv', help='Path to the output file')

parser.add_argument('--top_k', type=int, default=5, help='Number of top matches to retrieve')

parser.add_argument('--gpt3_model', type=str, default='text-davinci-003', help='GPT3 model to use')
parser.add_argument('--top_p', type=float, default=0.6, help='Temperature to use for GPT3')


args = parser.parse_args()

print('###################')
print('Arguments:')

print('Training path: {}'.format(args.training_path))
print('Test path: {}'.format(args.test_path))
print('Output path: {}'.format(args.output_path))

print('Top K: {}'.format(args.top_k))

print('GPT3 model: {}'.format(args.gpt3_model))
print('Top P: {}'.format(args.top_p))

print('###################')


'''
Load the training data
'''
training_df = pd.read_csv(args.training_path)
print('Number of training examples: {}'.format(len(training_df)))


'''
Sentence transformer model
'''

sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

if torch.cuda.is_available():
	sentence_model.cuda()

def get_sentence_emb(input_text):
	return sentence_model.encode(input_text)


'''
Group by situation, thought and create a list of reframes
'''
training_grouped_df = training_df.groupby(['situation', 'thought'])['reframe'].apply(list).reset_index(name='reframe_list')

# create training ids
training_grouped_df['thought_record_id'] = training_grouped_df.index

training_thought_record_ids = training_grouped_df['thought_record_id'].tolist()
training_situation_and_thought_emb = training_grouped_df.apply(lambda x: get_sentence_emb(x.situation + ' ' + x.thought), axis=1)


'''
Function that returns the top K matched thought records from the training data
'''
def get_top_k_matches(curr_situation, curr_thought, K=5):
	curr_situation_and_thought_emb = sentence_model.encode(curr_situation + ' ' + curr_thought)

	situation_and_thought_scores = util.dot_score(curr_situation_and_thought_emb, training_situation_and_thought_emb)[0].cpu().tolist()
	situation_and_thought_score_pairs = list(zip(training_thought_record_ids, situation_and_thought_scores))
	situation_and_thought_score_pairs_sorted = sorted(situation_and_thought_score_pairs, key=lambda x: x[1], reverse=True)

	matched_thought_record_ids = [x[0] for x in situation_and_thought_score_pairs_sorted[:K]]

	shuffle(matched_thought_record_ids)

	return matched_thought_record_ids


'''
OpenAI API
'''
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")


test_df = pd.read_csv(args.test_path)
print('Number of test examples: {}'.format(len(test_df)))

test_df['generated_reframe'] = ''

print('###################')
print('Generating reframes for the test data...')

for index, row in tqdm(test_df.iterrows(), total=len(test_df)):
	curr_situation = row['situation']
	curr_thought = row['thought']
	
	mathed_thought_record_ids_li = get_top_k_matches(curr_situation, curr_thought, K=args.top_k)

	matched_user_response_df = training_grouped_df[training_grouped_df['thought_record_id'].isin(mathed_thought_record_ids_li)]

	'''
	Create Prompt
	'''
	curr_retrieval_prompt = ''

	for inner_index, inner_row in matched_user_response_df.iterrows():
		# Sample a reframe from the reframe list
		curr_reframe = np.random.choice(inner_row['reframe_list'])

		curr_retrieval_prompt += 'Situation: ' + inner_row['situation'] + '\nDistorted Thought: ' + inner_row['thought'] + '\nRational Response: ' + curr_reframe + '\n\n'


	curr_test_input = 'Situation: ' + curr_situation.strip() + '\nDistorted Thought: ' + curr_thought.strip()

	'''
	Generate the rational response
	'''
	MAX_RETRIES = 5
	current_tries = 1

	while current_tries <= MAX_RETRIES:
		try:
			curr_response_reframing = openai.Completion.create(
				engine=args.gpt3_model,
				prompt=curr_retrieval_prompt + '\n\n' + curr_test_input + '\nRational Response:',
				max_tokens=256,
				top_p=args.top_p,
				frequency_penalty=0.0,
				presence_penalty=0.0,
				logprobs=5,
				stop=['\n'],
			)
			break
		except Exception as e:
			print('Error: {}'.format(e))	
			print('retrying')
			time.sleep(5)
			current_tries += 1
	
	curr_response_reframing_str = curr_response_reframing['choices'][0]['text'].replace('\n', ' ').strip()

	test_df.at[index, 'generated_reframe'] = curr_response_reframing_str

test_df.to_csv(args.output_path, index=False)

print('Output saved to: {}'.format(args.output_path))
print('###################')