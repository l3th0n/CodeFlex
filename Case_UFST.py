import requests, json

# test_case = {'points': [[10, 0], [10, 0], [10, 0], [8, 2], [10, 0], [2, 3], [7, 3], [10, 0], [10, 0], [10, 0], [10, 10]], 'token': 'G6lCvLlNHmLE16Fz9nYvZMfUfaTudJTL'}
g = requests.get('http://13.74.31.101/api/points')
bowling_dct = json.loads(g.text)
pointboard = bowling_dct['points']
token = bowling_dct['token']

scoreboard = [0] * len(pointboard)
i = 0
tot_score = 0
# print(bowling_dct)
# print(pointboard)

while i < len(pointboard): # Iterate over point list and calculate fowardly the scores of strikes, spares and regular rounds.
    score = 0
    if pointboard[i][0] == 10: # Strike - Search/look ahead 2 bowls (two scores)
        score = 10
        ii = i + 1
        while ii < len(pointboard) and ii < i + 3: # Make sure not to exceed indices.
            if pointboard[ii][0] == 10: # Another strike
                score += 10
                ii += 1
                continue # Increase index (ii) and continue loop
            if score == 20: score += pointboard[ii][0] # 2 (and not 3) strikes means add next rounds point
            else: score += pointboard[ii][0] + pointboard[ii][1] # Single strike, therefore add the two shots
            break
    elif sum(pointboard[i]) == 10: # Spare - Search/look ahead 1 bowl (1 score)
        if i + 1 < len(pointboard):
            score = 10 + pointboard[i+1][0]
        else: score = sum(pointboard[i])
    else: score = sum(pointboard[i]) # Regular round

    tot_score += score
    scoreboard[i] = tot_score
    i += 1

# print(scoreboard)
upload = {'token': token, 'points': scoreboard}
p = requests.post('https://httpbin.org/post', data=upload)
# print(p.text)
# print(p.status_code)
