def get_chunks(data, chunk_size=5164):
    chunks = []
    data_length = len(data)
    if data_length <= chunk_size:
        chunks.append(data)
    else:
        chunk_start = 0
        while chunk_start < data_length - chunk_size:
            chunk_end = chunk_start + chunk_size
            chunk = data[chunk_start:chunk_end]
            chunks.append(chunk)
            chunk_start += chunk_size - 200  # 每次滑动窗口向前移动200个字
        # 处理最后一个文本块
        last_chunk = data[chunk_start:]
        if len(last_chunk) == chunk_size:
            chunks.append(last_chunk)
        elif len(last_chunk) < chunk_size:
            last_chunk = chunks[-1][-((chunk_size - len(last_chunk))+1):-1] + last_chunk
            chunks.append(last_chunk)
    #print(chunks)
    return chunks

# 示例用法
#text = "为有效防范化解森林草原火灾风险，全力维护人民群众生命财产和生态安全."
#get_chunks(text, chunk_size=5164)
#print(result_chunks)