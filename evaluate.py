def cost_function(y_true, y_pred, scorings: list) -> dict:
    avg_result = 0
    scoring_results = dict()
    for score in scorings:
        if not bool(score.get('paramas')):
            score['params'] = {}

        score_result = score['metric'](y_true, y_pred, **score['params'])
        scoring_results[score['metric'].__name__] = score_result
        avg_result += score_result
        
    avg_result = avg_result/len(scoring_results)
    
    return avg_result, scoring_results