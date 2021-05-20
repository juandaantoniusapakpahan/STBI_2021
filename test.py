from lambdamart import LambdaMART
import numpy as np
import pandas as pd

def get_data(file_loc):
	f = open(file_loc, 'r')
	data = []
	for line in f:
		new_arr = []
		arr = line.split(' #')[0].split()
		score = arr[0]
		q_id = arr[1].split(':')[1]
		new_arr.append(int(score))
		new_arr.append(int(q_id))
		arr = arr[2:]
		for el in arr:
			new_arr.append(float(el.split(':')[1]))
		data.append(new_arr)
	f.close()
	return np.array(data)

def group_queries(data):
	query_indexes = {}
	index = 0
	for record in data:
		query_indexes.setdefault(record[1], [])
		query_indexes[record[1]].append(index)
		index += 1
	return query_indexes


def main():
	total_ndcg = 0.0
	for i in [1,2,3,4,5]:
		print ('start Fold ' + str(i))
		training_data = get_data('MQ2008/Fold%d/train.txt' % (i))
		test_data = get_data('MQ2008/Fold%d/test.txt' % (i))
		model = LambdaMART(training_data, 300, 0.001, 'sklearn')
		model.fit()
		model.save('lambdamart_model_%d' % (i))
		# model = LambdaMART()
		# model.load('lambdamart_model.lmart')
		average_ndcg, predicted_scores = model.validate(test_data, 10)
		print (average_ndcg)
		total_ndcg += average_ndcg
	total_ndcg /= 5.0
	print ('Original average ndcg at 10 is: ' + str(total_ndcg))


if __name__ == '__main__':
	main()
