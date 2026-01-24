#!/usr/bin/env python3
"""
Puzzle Evaluation Script - Evaluates CNN and Transformer models on Lichess puzzles

For each puzzle:
- Runs both models move by move
- Continues only if the prediction is correct
- Tracks completion percentage (how far through the puzzle each model got)
- Also tracks top-3 and top-5 completion (if correct move was in top 3 or 5)

Output columns:
- cnn_puzzle_pct: percentage of puzzle moves CNN predicted correctly (sequential)
- transformer_puzzle_pct: percentage of puzzle moves Transformer predicted correctly (sequential)
- cnn_puzzle_pct_top3: percentage when considering top-3 predictions
- cnn_puzzle_pct_top5: percentage when considering top-5 predictions
- transformer_puzzle_pct_top3: percentage when considering top-3 predictions
- transformer_puzzle_pct_top5: percentage when considering top-5 predictions
- cnn_moves_correct: number of consecutive correct moves by CNN
- transformer_moves_correct: number of consecutive correct moves by Transformer
- total_puzzle_moves: total number of moves the player needs to make in the puzzle
"""

import argparse
import sys
import torch
import chess
import pandas as pd
import ast
from pathlib import Path
from tqdm import tqdm

# Import CNN components
from cnn_policy.model import ChessCNNPolicy
from cnn_policy.position_encoder import PositionEncoder

# Import Transformer components
from transformer_policy.model import ChessTransformer
from transformer_policy.config import TransformerConfig


def load_cnn_model(checkpoint_path: str, device: torch.device):
    """Load CNN model from checkpoint. Uses architecture from cnn_policy/config.py."""
    # Architecture matches cnn_policy/config.py TrainingConfig
    model = ChessCNNPolicy(
        num_input_channels=18,  # Config.NUM_INPUT_CHANNELS
        num_filters=256,        # Config.NUM_FILTERS
        num_blocks=15,          # Config.NUM_BLOCKS
        dropout_rate=0.1        # Config.DROPOUT_RATE
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model


def load_transformer_model(checkpoint_path: str, device: torch.device):
    """Load Transformer model from checkpoint. Uses architecture from transformer_policy/config.py."""
    config = TransformerConfig()
    model = ChessTransformer(config)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model


def encode_board_transformer(board: chess.Board, device: torch.device) -> dict:
    """
    Encode chess board to transformer input format.
    Matches the piece encoding from cnn_policy/dataset_loader.py decode_board_from_h5.
    """
    # Piece encoding from dataset_loader.py:
    # 0 = empty
    # Even numbers (2,4,6,8,10,12) = BLACK pieces
    # Odd numbers (3,5,7,9,11,13) = WHITE pieces
    # Order: P, R, N, B, Q, K
    piece_map = {
        (chess.PAWN, chess.BLACK): 2,
        (chess.PAWN, chess.WHITE): 3,
        (chess.ROOK, chess.BLACK): 4,
        (chess.ROOK, chess.WHITE): 5,
        (chess.KNIGHT, chess.BLACK): 6,
        (chess.KNIGHT, chess.WHITE): 7,
        (chess.BISHOP, chess.BLACK): 8,
        (chess.BISHOP, chess.WHITE): 9,
        (chess.QUEEN, chess.BLACK): 10,
        (chess.QUEEN, chess.WHITE): 11,
        (chess.KING, chess.BLACK): 12,
        (chess.KING, chess.WHITE): 13,
    }
    
    board_positions = torch.zeros(64, dtype=torch.long)
    for square in range(64):
        piece = board.piece_at(square)
        if piece:
            board_positions[square] = piece_map[(piece.piece_type, piece.color)]
    
    return {
        'turns': torch.tensor([[0 if board.turn == chess.WHITE else 1]], dtype=torch.long).to(device),
        'white_kingside_castling_rights': torch.tensor(
            [[int(board.has_kingside_castling_rights(chess.WHITE))]], dtype=torch.long
        ).to(device),
        'white_queenside_castling_rights': torch.tensor(
            [[int(board.has_queenside_castling_rights(chess.WHITE))]], dtype=torch.long
        ).to(device),
        'black_kingside_castling_rights': torch.tensor(
            [[int(board.has_kingside_castling_rights(chess.BLACK))]], dtype=torch.long
        ).to(device),
        'black_queenside_castling_rights': torch.tensor(
            [[int(board.has_queenside_castling_rights(chess.BLACK))]], dtype=torch.long
        ).to(device),
        'board_positions': board_positions.unsqueeze(0).to(device),
    }


def get_top_k_moves_cnn(model, board: chess.Board, encoder: PositionEncoder, device: torch.device, k: int = 5):
    """
    Get top-k predicted moves from CNN model.
    Based on cnn_policy/inference.py predict_move function.
    """
    import torch.nn.functional as F
    
    model.eval()
    with torch.no_grad():
        # Encode position - matches inference.py
        position = encoder.fen_to_tensor(board.fen())
        position = position.unsqueeze(0).to(device)  # (1, 18, 8, 8)
        
        # Get predictions
        from_logits, to_logits = model(position)
        
        # Convert to log probabilities - matches inference.py
        from_log_probs = F.log_softmax(from_logits, dim=-1).unsqueeze(2)  # (1, 64, 1)
        to_log_probs = F.log_softmax(to_logits, dim=-1).unsqueeze(1)      # (1, 1, 64)
        
        # Combine: log P(from) + log P(to) = log P(from, to)
        combined = (from_log_probs + to_log_probs).view(1, -1)  # (1, 4096)
        
        # Get legal moves - remove promotion suffixes for matching
        legal_moves = [move.uci() for move in board.legal_moves]
        legal_moves_no_promotion = list(set([m[:4] for m in legal_moves]))
        
        # Convert legal moves to indices - matches inference.py SQUARES mapping
        legal_indices = []
        for move_str in legal_moves_no_promotion:
            from_idx = chess.SQUARE_NAMES.index(move_str[:2])
            to_idx = chess.SQUARE_NAMES.index(move_str[2:4])
            move_idx = from_idx * 64 + to_idx
            legal_indices.append(move_idx)
        
        # Filter predictions to legal moves only
        legal_scores = combined[0, legal_indices]
        
        # Get top-k
        actual_k = min(k, len(legal_indices))
        topk_scores, topk_indices = torch.topk(legal_scores, actual_k)
        
        top_moves = []
        for idx in topk_indices:
            move_str = legal_moves_no_promotion[idx.item()]
            
            # Handle pawn promotion - always queen (matches inference.py is_pawn_promotion logic)
            from_sq = chess.SQUARE_NAMES.index(move_str[:2])
            to_sq = chess.SQUARE_NAMES.index(move_str[2:4])
            piece = board.piece_at(from_sq)
            
            if piece and piece.piece_type == chess.PAWN:
                to_rank = to_sq // 8
                if (piece.color == chess.WHITE and to_rank == 7) or \
                   (piece.color == chess.BLACK and to_rank == 0):
                    move_str = move_str + "q"
            
            top_moves.append(move_str)
        
        return top_moves


def get_top_k_moves_transformer(model, board: chess.Board, device: torch.device, k: int = 5):
    """
    Get top-k predicted moves from Transformer model.
    Based on transformer_policy/inference.py predict_move function.
    """
    import torch.nn.functional as F
    
    model.eval()
    with torch.no_grad():
        # Encode board to transformer input format
        batch = encode_board_transformer(board, device)
        
        # Get predictions
        from_logits, to_logits = model(batch)
        from_logits = from_logits.squeeze(1)  # (1, 64)
        to_logits = to_logits.squeeze(1)      # (1, 64)
        
        # Combine log probabilities - matches inference.py
        from_log_probs = F.log_softmax(from_logits, dim=-1).unsqueeze(2)  # (1, 64, 1)
        to_log_probs = F.log_softmax(to_logits, dim=-1).unsqueeze(1)      # (1, 1, 64)
        combined = (from_log_probs + to_log_probs).view(1, -1)  # (1, 4096)
        
        # Get legal moves - matches inference.py
        legal_moves = [move.uci() for move in board.legal_moves]
        legal_moves_no_promo = list(set([m[:4] for m in legal_moves]))
        
        # Map to indices
        legal_indices = []
        for move_str in legal_moves_no_promo:
            from_idx = chess.SQUARE_NAMES.index(move_str[:2])
            to_idx = chess.SQUARE_NAMES.index(move_str[2:4])
            legal_indices.append(from_idx * 64 + to_idx)
        
        # Filter to legal moves
        legal_scores = combined[0, legal_indices]
        
        # Get top-k
        actual_k = min(k, len(legal_indices))
        topk_scores, topk_indices = torch.topk(legal_scores, actual_k)
        
        top_moves = []
        for idx in topk_indices:
            move_str = legal_moves_no_promo[idx.item()]
            
            # Handle promotions (always queen) - matches inference.py
            from_sq = chess.SQUARE_NAMES.index(move_str[:2])
            to_sq = chess.SQUARE_NAMES.index(move_str[2:4])
            piece = board.piece_at(from_sq)
            
            if piece and piece.piece_type == chess.PAWN:
                to_rank = to_sq // 8
                if (piece.color == chess.WHITE and to_rank == 7) or \
                   (piece.color == chess.BLACK and to_rank == 0):
                    move_str = move_str + "q"
            
            top_moves.append(move_str)
        
        return top_moves


def normalize_move(move_str: str) -> str:
    """Normalize move string for comparison (strip promotion piece for matching)."""
    return move_str[:4].lower()


def move_matches(predicted: str, expected: str) -> bool:
    """Check if predicted move matches expected move."""
    return normalize_move(predicted) == normalize_move(expected)


def move_in_top_k(top_k_moves: list, expected: str) -> bool:
    """Check if expected move is in top-k predictions."""
    expected_norm = normalize_move(expected)
    return any(normalize_move(m) == expected_norm for m in top_k_moves)


def evaluate_puzzle(fen: str, moves_str: str, 
                    cnn_model, transformer_model, 
                    encoder, device):
    """
    Evaluate a single puzzle with both models.
    
    In Lichess puzzles:
    - First move is the opponent's move that creates the puzzle position
    - Then alternating: player move, opponent response, player move, etc.
    - Player needs to find moves at indices 1, 3, 5, ... (0-indexed)
    
    Returns dict with evaluation results.
    """
    try:
        board = chess.Board(fen)
        moves = moves_str.split()
        
        if len(moves) < 2:
            return None
        
        # Apply the first move (opponent's move that sets up the puzzle)
        first_move = chess.Move.from_uci(moves[0])
        board.push(first_move)
        
        # Player moves are at odd indices (1, 3, 5, ...)
        player_move_indices = list(range(1, len(moves), 2))
        total_player_moves = len(player_move_indices)
        
        if total_player_moves == 0:
            return None
        
        # Track progress for each model
        cnn_correct_count = 0
        cnn_correct_top3_count = 0
        cnn_correct_top5_count = 0
        transformer_correct_count = 0
        transformer_correct_top3_count = 0
        transformer_correct_top5_count = 0
        
        cnn_stopped = False
        cnn_stopped_top3 = False
        cnn_stopped_top5 = False
        transformer_stopped = False
        transformer_stopped_top3 = False
        transformer_stopped_top5 = False
        
        # Simulate puzzle for both models
        cnn_board = board.copy()
        transformer_board = board.copy()
        cnn_board_top3 = board.copy()
        cnn_board_top5 = board.copy()
        transformer_board_top3 = board.copy()
        transformer_board_top5 = board.copy()
        
        for i, move_idx in enumerate(player_move_indices):
            expected_move = moves[move_idx]
            
            # Get opponent's response if there is one
            opponent_response_idx = move_idx + 1
            has_opponent_response = opponent_response_idx < len(moves)
            
            # --- CNN exact match evaluation ---
            if not cnn_stopped:
                try:
                    cnn_top_moves = get_top_k_moves_cnn(cnn_model, cnn_board, encoder, device, k=5)
                    cnn_predicted = cnn_top_moves[0] if cnn_top_moves else ""
                    
                    if move_matches(cnn_predicted, expected_move):
                        cnn_correct_count += 1
                        # Apply move and opponent response
                        cnn_board.push(chess.Move.from_uci(expected_move))
                        if has_opponent_response:
                            cnn_board.push(chess.Move.from_uci(moves[opponent_response_idx]))
                    else:
                        cnn_stopped = True
                except Exception as e:
                    cnn_stopped = True
            
            # --- CNN top-5 evaluation ---
            if not cnn_stopped_top5:
                try:
                    cnn_top_moves = get_top_k_moves_cnn(cnn_model, cnn_board_top5, encoder, device, k=5)
                    
                    if move_in_top_k(cnn_top_moves, expected_move):
                        cnn_correct_top5_count += 1
                        # Apply move and opponent response
                        cnn_board_top5.push(chess.Move.from_uci(expected_move))
                        if has_opponent_response:
                            cnn_board_top5.push(chess.Move.from_uci(moves[opponent_response_idx]))
                    else:
                        cnn_stopped_top5 = True
                except Exception as e:
                    cnn_stopped_top5 = True
            
            # --- CNN top-3 evaluation ---
            if not cnn_stopped_top3:
                try:
                    cnn_top_moves = get_top_k_moves_cnn(cnn_model, cnn_board_top3, encoder, device, k=3)
                    
                    if move_in_top_k(cnn_top_moves, expected_move):
                        cnn_correct_top3_count += 1
                        # Apply move and opponent response
                        cnn_board_top3.push(chess.Move.from_uci(expected_move))
                        if has_opponent_response:
                            cnn_board_top3.push(chess.Move.from_uci(moves[opponent_response_idx]))
                    else:
                        cnn_stopped_top3 = True
                except Exception as e:
                    cnn_stopped_top3 = True
            
            # --- Transformer exact match evaluation ---
            if not transformer_stopped:
                try:
                    trans_top_moves = get_top_k_moves_transformer(transformer_model, transformer_board, device, k=5)
                    trans_predicted = trans_top_moves[0] if trans_top_moves else ""
                    
                    if move_matches(trans_predicted, expected_move):
                        transformer_correct_count += 1
                        # Apply move and opponent response
                        transformer_board.push(chess.Move.from_uci(expected_move))
                        if has_opponent_response:
                            transformer_board.push(chess.Move.from_uci(moves[opponent_response_idx]))
                    else:
                        transformer_stopped = True
                except Exception as e:
                    transformer_stopped = True
            
            # --- Transformer top-5 evaluation ---
            if not transformer_stopped_top5:
                try:
                    trans_top_moves = get_top_k_moves_transformer(transformer_model, transformer_board_top5, device, k=5)
                    
                    if move_in_top_k(trans_top_moves, expected_move):
                        transformer_correct_top5_count += 1
                        # Apply move and opponent response
                        transformer_board_top5.push(chess.Move.from_uci(expected_move))
                        if has_opponent_response:
                            transformer_board_top5.push(chess.Move.from_uci(moves[opponent_response_idx]))
                    else:
                        transformer_stopped_top5 = True
                except Exception as e:
                    transformer_stopped_top5 = True
            
            # --- Transformer top-3 evaluation ---
            if not transformer_stopped_top3:
                try:
                    trans_top_moves = get_top_k_moves_transformer(transformer_model, transformer_board_top3, device, k=3)
                    
                    if move_in_top_k(trans_top_moves, expected_move):
                        transformer_correct_top3_count += 1
                        # Apply move and opponent response
                        transformer_board_top3.push(chess.Move.from_uci(expected_move))
                        if has_opponent_response:
                            transformer_board_top3.push(chess.Move.from_uci(moves[opponent_response_idx]))
                    else:
                        transformer_stopped_top3 = True
                except Exception as e:
                    transformer_stopped_top3 = True
        
        return {
            'cnn_moves_correct': cnn_correct_count,
            'cnn_moves_correct_top3': cnn_correct_top3_count,
            'cnn_moves_correct_top5': cnn_correct_top5_count,
            'transformer_moves_correct': transformer_correct_count,
            'transformer_moves_correct_top3': transformer_correct_top3_count,
            'transformer_moves_correct_top5': transformer_correct_top5_count,
            'total_puzzle_moves': total_player_moves,
            'cnn_puzzle_pct': round(100.0 * cnn_correct_count / total_player_moves, 2),
            'cnn_puzzle_pct_top3': round(100.0 * cnn_correct_top3_count / total_player_moves, 2),
            'cnn_puzzle_pct_top5': round(100.0 * cnn_correct_top5_count / total_player_moves, 2),
            'transformer_puzzle_pct': round(100.0 * transformer_correct_count / total_player_moves, 2),
            'transformer_puzzle_pct_top3': round(100.0 * transformer_correct_top3_count / total_player_moves, 2),
            'transformer_puzzle_pct_top5': round(100.0 * transformer_correct_top5_count / total_player_moves, 2),
        }
        
    except Exception as e:
        print(f"Error evaluating puzzle: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Evaluate models on Lichess puzzles')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file with puzzles')
    parser.add_argument('--output', type=str, default=None, help='Output CSV file (default: input_evaluated.csv)')
    parser.add_argument('--cnn-checkpoint', type=str, default='cnn_policy/checkpoints/checkpoint_step_55000.pth')
    parser.add_argument('--transformer-checkpoint', type=str, default='transformer_policy/checkpoints/checkpoint_step_55000.pth')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'])
    parser.add_argument('--limit', type=int, default=None, help='Limit number of puzzles to evaluate')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load models
    print("Loading CNN model...")
    cnn_model = load_cnn_model(args.cnn_checkpoint, device)
    encoder = PositionEncoder()
    
    print("Loading Transformer model...")
    transformer_model = load_transformer_model(args.transformer_checkpoint, device)
    
    # Load dataset
    print(f"Loading puzzles from {args.input}...")
    df = pd.read_csv(args.input)
    
    if args.limit:
        df = df.head(args.limit)
    
    print(f"Evaluating {len(df)} puzzles...")
    
    # Initialize new columns
    new_cols = [
        'cnn_moves_correct', 'cnn_moves_correct_top3', 'cnn_moves_correct_top5',
        'transformer_moves_correct', 'transformer_moves_correct_top3', 'transformer_moves_correct_top5',
        'total_puzzle_moves',
        'cnn_puzzle_pct', 'cnn_puzzle_pct_top3', 'cnn_puzzle_pct_top5',
        'transformer_puzzle_pct', 'transformer_puzzle_pct_top3', 'transformer_puzzle_pct_top5'
    ]
    for col in new_cols:
        df[col] = None
    
    # Evaluate each puzzle
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating puzzles"):
        result = evaluate_puzzle(
            row['FEN'],
            row['Moves'],
            cnn_model,
            transformer_model,
            encoder,
            device
        )
        
        if result:
            for col in new_cols:
                df.at[idx, col] = result[col]
    
    # Set output path
    if args.output:
        output_path = args.output
    else:
        input_path = Path(args.input)
        output_path = str(input_path.parent / f"{input_path.stem}_evaluated{input_path.suffix}")
    
    # Save results
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")
    
    # Print summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    
    valid_df = df[df['total_puzzle_moves'].notna()]
    
    print(f"\nTotal puzzles evaluated: {len(valid_df)}")
    print(f"\nCNN Model:")
    print(f"  Average puzzle completion (exact): {valid_df['cnn_puzzle_pct'].mean():.2f}%")
    print(f"  Average puzzle completion (top-3): {valid_df['cnn_puzzle_pct_top3'].mean():.2f}%")
    print(f"  Average puzzle completion (top-5): {valid_df['cnn_puzzle_pct_top5'].mean():.2f}%")
    print(f"  Puzzles fully solved (exact): {(valid_df['cnn_puzzle_pct'] == 100).sum()} ({100*(valid_df['cnn_puzzle_pct'] == 100).mean():.2f}%)")
    print(f"  Puzzles fully solved (top-3): {(valid_df['cnn_puzzle_pct_top3'] == 100).sum()} ({100*(valid_df['cnn_puzzle_pct_top3'] == 100).mean():.2f}%)")
    print(f"  Puzzles fully solved (top-5): {(valid_df['cnn_puzzle_pct_top5'] == 100).sum()} ({100*(valid_df['cnn_puzzle_pct_top5'] == 100).mean():.2f}%)")
    
    print(f"\nTransformer Model:")
    print(f"  Average puzzle completion (exact): {valid_df['transformer_puzzle_pct'].mean():.2f}%")
    print(f"  Average puzzle completion (top-3): {valid_df['transformer_puzzle_pct_top3'].mean():.2f}%")
    print(f"  Average puzzle completion (top-5): {valid_df['transformer_puzzle_pct_top5'].mean():.2f}%")
    print(f"  Puzzles fully solved (exact): {(valid_df['transformer_puzzle_pct'] == 100).sum()} ({100*(valid_df['transformer_puzzle_pct'] == 100).mean():.2f}%)")
    print(f"  Puzzles fully solved (top-3): {(valid_df['transformer_puzzle_pct_top3'] == 100).sum()} ({100*(valid_df['transformer_puzzle_pct_top3'] == 100).mean():.2f}%)")
    print(f"  Puzzles fully solved (top-5): {(valid_df['transformer_puzzle_pct_top5'] == 100).sum()} ({100*(valid_df['transformer_puzzle_pct_top5'] == 100).mean():.2f}%)")


if __name__ == "__main__":
    main()
