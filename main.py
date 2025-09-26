import streamlit as st
import time

st.title('streamlit・超入門')

st.write('プレグレスバーの表示')
'Start!'

latest_iteration = st.empty()
bar = st.progress(0)

for i in range(100):
     latest_iteration.text(f'Iteration{i+1}')
     bar.progress(i + 1)
     time.sleep(0.1)

'Done!!!'
     
left_column, right_column = st.columns(2)
button = left_column.button('右カラムに文字を表示')
if button:
     right_column.write('ここは右カラムです。')

expander = st.expander('問い合わせ')
expander.write('問い合わせ')


# text = st.text_input('あなたの趣味を教えてください。')
# 'あなたの趣味は', text, 'です。'

# condition = st.slider('あなたの今は調子は？', 0, 10, 5)
# 'コンディション:', condition

# option = st.selectbox(
#     'あなたが好きな数字を教えてください。',
#     list(range(1, 11)))

# 'あなたの好きな数字は、', option,'です。'

# if st.checkbox('Show Image'):
#     img = Image.open('ChatGPT Image 2025年9月16日 15_28_08.png')
#     st.image(img, caption='cat')

# df = pd.DataFrame(
#     np.random.rand(100, 2)/[50, 50] + [35.69, 139.70],
#     columns=['lat', 'lon']
# )

# st.map(df)

# """
# # 章
# ## 節
# ### 項

# ```python
# import streamlit as st
# import numpy as np
# import pandas as pd
# ```
# """