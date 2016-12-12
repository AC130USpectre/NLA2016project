import numpy as np
import vk

session = vk.Session()
api = vk.API(session)

# First we need to count how many people are there
GROUP_ID = 'podslyshano_monch'

members_count = api.groups.getMembers(group_id='podslyshano_monch')['count']
print("Количество людей в сообществе:\t{}".format(members_count))

res = []
for i in range(members_count // 1000 + 1):
    res.append(api.groups.getMembers(
        group_id='podslyshano_monch', offset=i * 1000)['users'])
print("Собрали..")
members = [val for sublist in res for val in sublist]
del res

with open('export', 'w') as f:
    depth = 0
    depth_threshold = 2
    i_row = 0  # Current row for threshhold
    j_row = 0
    k_row = 0
    members_done = {}
    queue = members
    print("Initial step")
    for member in members:
        members_done[member] = True
        friends = api.friends.get(user_id=member)
        for friend in friends:
            f.writelines("{} {}\n".format(member, friend))
            queue.append(friend)
            k_row = k_row + 1
    i_row = k_row
    print("Next Steps")
    while queue:
        if j_row % 1000:
            print(i_row, j_row, k_row)
            if np.unique(queue).shape[0] >= 5 * 10**6:
                with open('status', 'w') as f_status:
                    f_status.writelines("Очередь слишком большая: {}, i ={}, j={}, k={}".format(
                        np.unique(queue).shape[0], i_row, j_row, k_row))
                    break

        val = queue.pop(0)  # Вытаскиваем id из очереди
        j_row = j_row + 1  # Увеличиваем счетчик проверенных id
        if j_row == i_row:  # Если достигли границы для этой глубины
            depth = depth + 1  # Увеличиваем показатель глубины
            print("Перешли на глубину:\t{}".format(depth))
            if depth == depth_threshold:  # Достигли лимита
                break
            # Пересчитываем границу исходя из текущей последней строчки
            i_row = k_row
        if val not in members_done.keys():  # Если еще не проверяли id
            members_done[val] = True  # Указываем что проверили
            try:
                # Получаем список друзей
                friends = api.friends.get(user_id=val)
                for friend in friends:
                    # Записываем все связи
                    f.writelines("{} {}\n".format(val, friend))
                    # На каждую запись увеличиваем счетчик строк
                    k_row = k_row + 1
                    queue.append(friend)  # Добавляем в очередь на проверку
            except:
                continue
