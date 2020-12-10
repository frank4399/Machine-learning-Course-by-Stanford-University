function [accuracy, precision, recall, F1_score] = Error_metrics(true_positive, ...
true_negative, false_positive, false_negative)

total_examples = true_positive+true_negative+false_positive+false_negative;

accuracy = (true_positive+true_negative)/(total_examples);

precision = true_positive/(true_positive+false_positive);

recall = true_positive/(true_positive+false_negative);

F1_score = (2*precision*recall)/(precision+recall);
end