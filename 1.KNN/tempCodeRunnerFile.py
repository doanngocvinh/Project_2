result = {
    'confidence': confidence,
    'accuracy_validation': accuracy,
    'accuracy_test': acc,
}


# Ghi các giá trị vào file văn bản
with open('../GUI/knn_evalute.txt', 'w') as file:
	for key, value in result.items():
		file.write(f"{round(value,3)}\n")
