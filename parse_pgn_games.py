"""
Parse PGN chess game files and extract game information to CSV.

This module parses PGN (Portable Game Notation) files and extracts game metadata
including moves, results, players, and other game information.
"""

import pandas as pd
import re
from typing import List, Dict


def count_moves(move_text: str) -> int:
    """
    Count the number of moves in a PGN move text string.
    
    In chess notation, a "move" refers to a full move (white + black).
    For example, "1. d4 Nf6" is one move (move 1), not two moves.
    
    Args:
        move_text: String containing chess moves in PGN notation
        
    Returns:
        Number of moves counted (full moves, not individual player moves)
    """
    if not move_text or not move_text.strip():
        return 0
    
    # Remove result indicators (1-0, 0-1, 1/2-1/2) at the end
    move_text_clean = re.sub(r'\s*(1-0|0-1|1/2-1/2)\s*$', '', move_text.strip())
    move_text_clean = move_text_clean.strip()
    
    if not move_text_clean:
        return 0
    
    # Count move numbers (like "1.", "2.", "50.", etc.)
    # Each move number represents one full move (white + black)
    # Pattern: number followed by period and optional space
    move_numbers = re.findall(r'\d+\.', move_text_clean)
    
    # The number of move numbers is the number of full moves
    num_moves = len(move_numbers)
    
    return num_moves


def parse_pgn_file(pgn_path: str) -> List[Dict]:
    """
    Parse a PGN file and extract game information.
    
    Args:
        pgn_path: Path to the PGN file
        
    Returns:
        List of dictionaries, each containing game metadata
    """
    games = []
    
    with open(pgn_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by pattern that matches: metadata block, blank line(s), moves, blank line(s)
    # Pattern: [metadata lines] followed by \n\n+ followed by [moves] followed by \n\n+
    # We'll split on triple newlines or more to separate games
    # But we need to handle each game as: metadata section + moves section
    
    # Split content into lines
    all_lines = content.split('\n')
    
    i = 0
    while i < len(all_lines):
        # Skip empty lines at the start
        while i < len(all_lines) and not all_lines[i].strip():
            i += 1
        
        if i >= len(all_lines):
            break
        
        # Extract metadata tags (lines starting with [)
        metadata = {}
        while i < len(all_lines):
            line = all_lines[i].strip()
            if not line:
                i += 1
                break  # Blank line separates metadata from moves
            
            # Check if this is a metadata tag
            match = re.match(r'\[(\w+)\s+"([^"]+)"\]', line)
            if match:
                key, value = match.groups()
                metadata[key] = value
                i += 1
            else:
                # Not a metadata tag, might be start of moves or blank line
                if line:
                    # This might be moves, but we'll handle it after the blank line
                    break
                i += 1
        
        # Skip if no metadata found (not a valid game)
        if not metadata:
            continue
        
        # Skip blank lines to get to moves
        while i < len(all_lines) and not all_lines[i].strip():
            i += 1
        
        # Now collect move text (until we hit another blank line or metadata tag)
        move_text = ""
        while i < len(all_lines):
            line = all_lines[i].strip()
            
            # If we hit a blank line and we already have moves, we're done with this game
            if not line:
                if move_text.strip():
                    break
                i += 1
                continue
            
            # If we hit a new metadata tag, we're done with this game
            if re.match(r'\[(\w+)\s+"([^"]+)"\]', line):
                break
            
            # This is part of the move text
            move_text += " " + line
            i += 1
        
        # Count moves in the move text
        num_moves = count_moves(move_text)
        
        # Determine winner based on result
        result = metadata.get('Result', '')
        white_player = metadata.get('White', '')
        black_player = metadata.get('Black', '')
        winner = None
        
        if result == '1-0':
            winner = white_player
        elif result == '0-1':
            winner = black_player
        elif result == '1/2-1/2':
            winner = 'Draw'
        else:
            winner = 'Unknown'
        
        # Create game record
        game_record = {
            'game_id': metadata.get('GameId', ''),
            'date': metadata.get('Date', ''),
            'utc_date': metadata.get('UTCDate', ''),
            'utc_time': metadata.get('UTCTime', ''),
            'white_player': white_player,
            'black_player': black_player,
            'white_elo': metadata.get('WhiteElo', ''),
            'black_elo': metadata.get('BlackElo', ''),
            'white_rating_diff': metadata.get('WhiteRatingDiff', ''),
            'black_rating_diff': metadata.get('BlackRatingDiff', ''),
            'result': result,
            'winner': winner,
            'num_moves': num_moves,
            'time_control': metadata.get('TimeControl', ''),
            'eco': metadata.get('ECO', ''),
            'termination': metadata.get('Termination', ''),
            'variant': metadata.get('Variant', ''),
            'site': metadata.get('Site', '')
        }
        
        games.append(game_record)
    
    return games


def create_games_csv(pgn_path: str, output_csv_path: str) -> pd.DataFrame:
    """
    Parse a PGN file and create a CSV file with game summaries.
    
    Args:
        pgn_path: Path to the input PGN file
        output_csv_path: Path to the output CSV file
        
    Returns:
        DataFrame containing the game data
    """
    # Parse the PGN file
    games_data = parse_pgn_file(pgn_path)
    
    # Convert to DataFrame
    games_df = pd.DataFrame(games_data)
    
    # Save to CSV
    games_df.to_csv(output_csv_path, index=False)
    
    return games_df


if __name__ == '__main__':
    # Example usage
    pgn_path = 'Data/lichess_AuraChessTransformer_2026-01-22.pgn'
    output_csv_path = 'Data/games_summary.csv'
    
    print(f"Parsing PGN file: {pgn_path}")
    games_df = create_games_csv(pgn_path, output_csv_path)
    
    print(f"\nTotal games parsed: {len(games_df)}")
    print(f"Games DataFrame shape: {games_df.shape}")
    print(f"\nFirst few games:")
    print(games_df.head(10))
    
    print(f"\n✓ Saved game summary to: {output_csv_path}")
    
    # Display move statistics
    print(f"\nMove statistics:")
    print(f"  Average number of moves: {games_df['num_moves'].mean():.2f}")
    print(f"  Median number of moves: {games_df['num_moves'].median():.2f}")
    print(f"  Min moves: {games_df['num_moves'].min()}")
    print(f"  Max moves: {games_df['num_moves'].max()}")
    print(f"  Games with 0 moves: {(games_df['num_moves'] == 0).sum()}")
