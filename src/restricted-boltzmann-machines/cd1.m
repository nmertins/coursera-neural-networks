function ret = cd1(rbm_w, visible_data)
% <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
% <visible_data> is a (possibly but not necessarily binary) matrix of size <number of visible units> by <number of data cases>
% The returned value is the gradient approximation produced by CD-1. It's of the same shape as <rbm_w>.

	visible_data = sample_bernoulli(visible_data);

    hidden_state_0 = generate_hidden_state(rbm_w, visible_data);
   	configuration_goodness_grad_0 = configuration_goodness_gradient(visible_data, hidden_state_0);

   	reconstruction = generate_visible_state(rbm_w, hidden_state_0);

   	hidden_state_1 = generate_hidden_state(rbm_w, reconstruction, true);
   	configuration_goodness_grad_1 = configuration_goodness_gradient(reconstruction, hidden_state_1);

   	ret = configuration_goodness_grad_0 - configuration_goodness_grad_1;
end

function hidden_state = generate_hidden_state(rbm_w, visible_data, skip_sample = false)
	hidden_probabilities = visible_state_to_hidden_probabilities(rbm_w, visible_data);
	if(skip_sample)
		hidden_state = hidden_probabilities;
	else
	   	hidden_state = sample_bernoulli(hidden_probabilities);
	endif
end

function visible_state = generate_visible_state(rbm_w, hidden_state)
	visible_probabilities = hidden_state_to_visible_probabilities(rbm_w, hidden_state);
	visible_state = sample_bernoulli(visible_probabilities);
end