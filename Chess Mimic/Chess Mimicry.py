import requests
import numpy as np
import re
import chess
import tensorflow as tf

headers = {'User-Agent': 'Mozilla/5.0'}
MAX_POSITIONS = 1000000
BATCH_SIZE = 10000  # batch size for incremental processing

list_of_games = []
X_tensors, y_labels = [], []
move_to_index, index_to_move = {}, {}

piece_to_index = {
    '.': 0, 'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
    'p': 7, 'n': 8, 'b': 9, 'r': 10, 'q': 11, 'k': 12
}

def board_to_tensor(board):
    tensor = np.zeros((8, 8, 13), dtype=np.float32)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        symbol = piece.symbol() if piece else '.'
        row = 7 - (square // 8)  # chess library square numbering: 0=a1, 63=h8
        col = square % 8
        tensor[row][col][piece_to_index[symbol]] = 1.0
    return tensor

def add_game(moves):
    global X_tensors, y_labels
    board = chess.Board()
    tokens = moves.split()
    i = 0
    move_dict = {}

    while i < len(tokens) and len(X_tensors) < MAX_POSITIONS:
        if tokens[i].endswith('.'):
            try:
                move_num = int(tokens[i][:-1])
                white_move = tokens[i + 1]
                black_move = tokens[i + 2] if i + 2 < len(tokens) and not tokens[i + 2].endswith('.') else None
            except:
                i += 1
                continue

            move_dict[move_num] = [white_move.strip(), black_move.strip() if black_move else None]

            try:
                board.push_san(white_move)
                features = board_to_tensor(board)
                if black_move:
                    X_tensors.append(features)
                    if black_move not in move_to_index:
                        idx = len(move_to_index)
                        move_to_index[black_move] = idx
                        index_to_move[idx] = black_move
                    y_labels.append(move_to_index[black_move])
                    board.push_san(black_move)
            except:
                break

            i += 3
        else:
            i += 1

    if move_dict:
        list_of_games.append(move_dict)

def clean_pgn(pgn):
    tokens = pgn.split()
    return ' '.join([token for token in tokens if not token.endswith('...')])

def get_game_data(username):
    archive_response = requests.get(f"https://api.chess.com/pub/player/{username}/games/archives", headers=headers)
    if not archive_response.ok:
        raise Exception(f'Could not retrieve archive data for {username}')

    count_games = 0
    for archive in archive_response.json()["archives"]:
        game_response = requests.get(archive, headers=headers)
        if not game_response.ok:
            continue
        for game in game_response.json()["games"]:
            try:
                pgn_file = game["pgn"]
            except:
                continue
            if f'[White "{username}"]' in pgn_file:
                continue
            if "Chess960" in pgn_file:
                continue
            pgn_clean = re.sub(r"\{[^}]*\}", "", pgn_file)
            moves_section = pgn_clean.split("\n\n")[-1]
            moves_section = re.sub(r"\s+", " ", moves_section).strip()
            add_game(clean_pgn(moves_section))
            count_games += 1
            if len(X_tensors) >= MAX_POSITIONS:
                print(f"Reached max positions with {count_games} games.")
                return
    print(f"Collected {count_games} games in total.")

def train_model_tf():
    if not y_labels:
        print("No training data collected.")
        return None

    # Convert lists to numpy arrays in batches to avoid memory issues
    X_arr = np.array(X_tensors)
    y_arr = tf.keras.utils.to_categorical(y_labels, num_classes=len(move_to_index))

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(8, 8, 13)),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(len(move_to_index), activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(X_arr, y_arr, epochs=5, batch_size=64, verbose=1)
    print(f"Trained on {len(y_labels)} positions.")
    return model

def predict_move_tf(board, model):
    features = board_to_tensor(board).reshape(1, 8, 8, 13)
    preds = model.predict(features, verbose=0)
    top_index = np.argmax(preds[0])
    move_san = index_to_move.get(top_index)
    for move in board.legal_moves:
        if board.san(move) == move_san:
            return move_san
    return None

def find_exact_move(move, count, current_game):
    move = move.strip()
    for move_list in list_of_games:
        if count in move_list and move_list[count][0].strip() == move:
            match = True
            for key in current_game:
                if key not in move_list:
                    match = False
                    break
                stored_white = move_list[key][0].strip() if move_list[key][0] else ''
                stored_black = move_list[key][1].strip() if move_list[key][1] else ''
                current_white = current_game[key][0].strip() if current_game[key][0] else ''
                current_black = current_game[key][1].strip() if len(current_game[key]) > 1 else ''

                if current_white == stored_white and (current_black == stored_black or current_black == ''):
                    continue
                else:
                    match = False
                    break
            if match:
                return move_list[count][1]
    return None

def lets_play(model):
    board = chess.Board()
    count = 1
    current_game = {}
    print("Start the game by entering your moves in SAN notation (e.g., e4, Nf3, d5)")
    while not board.is_game_over():
        print("\nCurrent board:")
        print(board)
        move = input("Your move: ").strip()
        try:
            board.push_san(move)
        except:
            print("Illegal move. Try again.")
            continue

        if count not in current_game:
            current_game[count] = [None]
        current_game[count][0] = move

        exact_move = find_exact_move(move, count, current_game)
        if exact_move and exact_move != 'None' and exact_move != '':
            print(f"Exact mimic move from DB: {exact_move}")
            board.push_san(exact_move)
            current_game[count].append(exact_move)
            count += 1
            continue

        ml_move = predict_move_tf(board, model)
        if ml_move:
            print(f"ML predicted move: {ml_move}")
            board.push_san(ml_move)
            if count not in current_game:
                current_game[count] = [move]
            if len(current_game[count]) == 1:
                current_game[count].append(ml_move)
            count += 1
        else:
            print("No valid mimic move found. AI skips its move.")
            count += 1

def main():
    username = input("Enter the username to mimic (playing as Black): ")
    print("Downloading and processing game data. This may take a few moments...")
    get_game_data(username)
    print(f"Collected {len(X_tensors)} positions for training.")
    model = train_model_tf()
    if model:
        lets_play(model)
    else:
        print("Insufficient data to train model. Exiting.")

if __name__ == "__main__":
    main()
