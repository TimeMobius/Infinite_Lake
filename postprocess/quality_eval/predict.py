from postprocess.quality_eval.qc_utils import (export_result, init_spark,
                                               load_dataset, predict,
                                               prepare_model)


def predict_score(dataset_path,
                  result_path,
                  model='gpt3',
                  tokenizer=None,
                  keep_method='gpt3',
                  text_key='text',
                  overall_stats=True):

    # set default tokenizers for default models
    if model == 'chinese':
        tokenizer = 'zh.sp.model'
        keep_method = 'label'
    if model == 'code':
        tokenizer = 'code.sp.model'
        keep_method = 'label'
    if model == 'gpt3':
        tokenizer = None
        keep_method = 'gpt3'

    # initialize a spark session
    if '_JAVA_OPTIONS' in os.environ and \
            '-Djava.net.preferIPv6Addresses=true' \
            in os.environ['_JAVA_OPTIONS']:
        os.environ['_JAVA_OPTIONS'] = os.environ['_JAVA_OPTIONS'].replace(
            '-Djava.net.preferIPv6Addresses=true',
            '-Djava.net.preferIPv6Addresses=false')
    spark = init_spark()
    # load the quality classifier model
    model = prepare_model(model_name=model)
    # load dataset
    ds = load_dataset(spark, dataset_path, text_key=text_key)
    # start to predict
    pred = predict(model, ds, tokenizer=tokenizer, keep_method=keep_method)
    # export prediction result to specific path
    export_result(pred, result_path)

    if overall_stats:
        # generate overall statistics on doc scores
        overall = pred.select('doc_score').toPandas().describe(include='all')
        # export to result report file
        overall.to_csv(os.path.join(result_path, 'overall.csv'))
        overall.to_markdown(os.path.join(result_path, 'overall.md'))
        return overall
