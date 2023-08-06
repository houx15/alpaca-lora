def sentence_cleaner(sentence:str):
    import re
    sentence = sentence.replace("\"", "")
    sentence = sentence.replace("RT", "")
    sentence = sentence.replace(".", "")
    sentence = sentence.replace("\'", "")
    results = re.compile(r'[http|https]*://[a-zA-Z0-9.?/&=:_%,-~]*', re.S)
    sentence = re.sub(results, '', sentence)
    sentence = re.sub('[\u4e00-\u9fa5]', '', sentence)
    # results2 = re.compile(r'[@].*?[ ]', re.S)
    # sentence = re.sub(results2, '', sentence)
    sentence = sentence.replace("\n", " ")
    sentence = sentence.strip()
    results2 = re.compile(r'[@].*?[ ]', re.S)
    sentence = re.sub(results2, '', sentence)
    return sentence

    
def result_translator(topic, result_text, translator):
    '''translate results to integer labels'''
    
    result_text = result_text.strip()
    result_text = result_text.strip('.')
    result_text = result_text.lower()
    result_text = result_text.replace('</s><s>', '')
    # for splitter in self.translator['splitters']:
    #     result_text = result_text.split(splitter)[0] + splitter
    #     if result_text in self.translator['labels'].keys():
    #         break
    print(f'result text: {result_text}')
    
    if result_text in translator['labels'].keys():
        return translator['labels'][result_text]
    elif 'irrelevant' in result_text:
        return -9
    elif 'neutral' in result_text:
        return 0
    elif 'slightly' in result_text:
        if topic == 'abortion':
            if 'illegal' in result_text:
                return -1
            else:
                return 1
        if topic == 'gun':
            if 'gun rights' in result_text:
                return -1
            else:
                return 1
        if topic == 'climate':
            if 'disagree' in result_text:
                return -1
            else:
                return 1
        if topic == 'sexual':
            if 'conservative' in result_text:
                return 1
            else:
                return -1
        if topic == 'drug':
            if 'prohibition' in result_text:
                return -1
            else:
                return 1
    elif 'strongly' in result_text:
        if topic == 'abortion':
            if 'illegal' in result_text:
                return -2
            else:
                return 2
        if topic == 'gun':
            if 'gun rights' in result_text:
                return -2
            else:
                return 2
        if topic == 'climate':
            if 'disagree' in result_text:
                return -2
            else:
                return 2
        if topic == 'sexual':
            if 'conservative' in result_text:
                return 2
            else:
                return -2
        if topic == 'drug':
            if 'prohibition' in result_text:
                return -2
            else:
                return 2
    return 0