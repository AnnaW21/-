import openai
# import time

openai.api_key = 'API_KEY'
client = openai.OpenAI(
    api_key= 'API_KEY',
)

def search(text):
  chat = client.chat.completions.create(model="gpt-3.5-turbo", messages=[ {"role": "user", "content": f'Выбери одно из представленных в списке ["Атмосфера", "Баланс", "Влияние", "Вызов", "Гибкость", "Деньги", "Динамика", "Достижение", "Инновации", "Интерес", "Карьера", "Командная работа", "Культура", "Люди", "Новизна", "Польза", "Признание", "Путешествия", "Развитие", "Репутация компании", "Руководство", "Свобода", "Стратегия", "Эмоции", "Самоутверждение", "Выгода", "Инфраструктура", "Общение", "Конфликты", "Страх", "Ценности", "Месть", "Вдохновение","Изоляция"] ключевых слов, подходящее по смыслу предложения : {text}'} ])
  reply = chat.choices[0].message.content
  return reply

# start_time = time.time()
text = list(df_rus['Ответ'])
final_lst = []
for string in text:
  reply = str(search(string))
  final_lst.append(reply)
# print("--- %s seconds ---" % (time.time() - start_time))
print(f"ChatGPT reply: {final_lst}")
