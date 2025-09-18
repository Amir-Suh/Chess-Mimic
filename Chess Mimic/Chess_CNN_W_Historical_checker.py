import requests
import numpy as np
import re
import chess
import chess.pgn as pgn
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
import random

headers = {'User-Agent': 'Mozilla/5.0'}
MAX_POSITIONS = 1000000

list_of_games = []
X_tensors, y_labels = [], []
move_to_index, index_to_move = {}, {}

# -----------------------------
# Encode board as 8x8x12 tensor
# -----------------------------
def board_to_tensor(board):
    matrix = np.zeros((8, 8, 12), dtype=np.float32)
    piece_map = board.piece_map()
    for square, piece in piece_map.items():
        row, col = divmod(square, 8)
        piece_type = piece.piece_type - 1  # piece_type is 1–6
        piece_color = 0 if piece.color else 6  # white = 0–5, black = 6–11
        matrix[row, col, piece_type + piece_color] = 1.0
    return matrix

# -----------------------------
# Add a PGN game to dataset
# -----------------------------
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

def clean_pgn(pgn_text):
    tokens = pgn_text.split()
    return ' '.join([token for token in tokens if not token.endswith('...')])

# -----------------------------
# Download games from Chess.com
# -----------------------------
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

# -----------------------------
# Train CNN model
# -----------------------------
def train_model_tf():
    if not y_labels:
        print("No training data collected.")
        return None

    X_arr = np.array(X_tensors)
    y_arr = to_categorical(y_labels, num_classes=len(move_to_index))

    model = Sequential([
        Conv2D(64, (3, 3), activation='relu', input_shape=(8, 8, 12)),
        Conv2D(128, (3, 3), activation='relu'),
        Flatten(),
        Dense(256, activation='relu'),
        Dense(len(move_to_index), activation='softmax')
    ])

    model.compile(optimizer=Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(X_arr, y_arr, epochs=20, batch_size=64, validation_split=0.1, verbose=1)
    print(f"Trained on {len(y_labels)} positions.")
    model.save("TF_CNN_Model.keras")
    return model

# -----------------------------
# Predict move with CNN
# -----------------------------
def predict_move_tf(board, model):
    features = board_to_tensor(board).reshape(1, 8, 8, 12)
    preds = model.predict(features, verbose=0)
    top_indices = np.argsort(preds[0])[::-1]

    for idx in top_indices:
        move_san = index_to_move.get(idx)
        for move in board.legal_moves:
            if board.san(move) == move_san:
                return move_san
    return None

# -----------------------------
# Check historical games for exact move (random choice if multiple)
# -----------------------------
def find_exact_move_random(board, current_game):
    move_number = len(current_game)
    matching_moves = []

    for game_dict in list_of_games:
        if move_number in game_dict:
            white_move, black_move = game_dict[move_number]
            temp_board = chess.Board()
            match = True
            for move_num in current_game:
                moves_played = current_game[move_num]
                for m in moves_played:
                    if m:
                        try:
                            temp_board.push_san(m)
                        except:
                            match = False
                            break
                if not match:
                    break
            if match and board.fen() == temp_board.fen():
                if black_move:
                    matching_moves.append(black_move)
    
    if matching_moves:
        return random.choice(matching_moves)
    return None

# -----------------------------
# Play a game loop with historical check + CNN
# -----------------------------
def lets_play(model):
    board = chess.Board()
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
        
        # Record the move in current game
        move_number = len(current_game) + 1
        if move_number not in current_game:
            current_game[move_number] = [move]
        else:
            current_game[move_number].append(move)
        
        exact_move = find_exact_move_random(board, current_game)
        if exact_move:
            try:
                board.push_san(exact_move)
                current_game[move_number].append(exact_move)
                print(f"Exact mimic move from historical games: {exact_move}")
            except:
                print("Historical move illegal, falling back to CNN.")
                cnn_move = predict_move_tf(board, model)
                if cnn_move:
                    board.push_san(cnn_move)
                    current_game[move_number].append(cnn_move)
                    print(f"Bot plays (CNN prediction): {cnn_move}")
                else:
                    print("No valid move predicted. Bot skips.")
        else:
            cnn_move = predict_move_tf(board, model)
            if cnn_move:
                board.push_san(cnn_move)
                current_game[move_number].append(cnn_move)
                print(f"Bot plays (CNN prediction): {cnn_move}")
            else:
                print("No valid move predicted. Bot skips.")

# -----------------------------
# Main entry
# -----------------------------
def main():
    username = input("Enter the username to mimic (playing as Black): ")
    print("Downloading and processing game data...")
    get_game_data(username)
    print(f"Collected {len(X_tensors)} positions for training.")
    model = train_model_tf()
    if model:
        lets_play(model)
    else:
        print("Insufficient data to train model. Exiting.")

if __name__ == "__main__":
    main()
