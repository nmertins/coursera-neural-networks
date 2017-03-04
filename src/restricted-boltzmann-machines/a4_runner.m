function a4_runner()
	learning_rates = [0.05, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0];

	for i = learning_rates
		printf('Results for learning rate: %d\n', i);
		a4_main(300, 0.02, i, 1000)
	endfor
end