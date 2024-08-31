% Data
predictions = [1 1 1 1 0 1 0 1 1 1 1 1 0 1 1 1 0 0 1 1 1 1 1 1 0];
actuals = [1 1 1 1 0 1 0 1 0 1 0 0 0 0 1 1 0 0 1 0 1 1 1 1 0];
correct = [true true true true true true true true false true false false true false true true false true true false true true true true true];

% Summarize correct and incorrect counts
correct_count = sum(correct);
incorrect_count = length(correct) - correct_count;

% Bar plot
bar([correct_count, incorrect_count]);
set(gca, 'XTickLabel', {'Correct', 'Incorrect'});
ylabel('Count');
title('Prediction Accuracy');