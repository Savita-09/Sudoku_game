import streamlit as st
import os
import time
#from dotenv import load_dotenv
import numpy as np
import matplotlib.pyplot as plt
import random

# Optional CrewAI imports
try:
    from crewai import Agent, Task, Crew
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False

load_dotenv(override=True)

st.set_page_config(
    page_title="Sudoku Master",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Optional LLM Config (for CrewAI)
# -----------------------------
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

def get_sudoku_explanation(board, solution):
    """
    Optional AI explanation using CrewAI.
    If CrewAI is unavailable, return fallback text.
    """
    if not CREWAI_AVAILABLE or not GROQ_API_KEY:
        return "Try scanning rows, columns, and 3x3 boxes to find the safest next move."

    try:
        sudoku_coach = Agent(
            role="Sudoku Coach",
            goal="Explain Sudoku solving steps clearly and briefly",
            backstory="You are an expert Sudoku tutor who gives practical hints.",
            verbose=False,
            allow_delegation=False
        )

        task = Task(
            description=f"""
You are given a Sudoku puzzle and its solution.

Current puzzle:
{board.tolist()}

Solved board:
{solution.tolist()}

Give ONE short helpful hint for the player without revealing too much.
Do not dump the full answer.
Keep it under 50 words.
""",
            expected_output="A short Sudoku hint."
        )

        crew = Crew(
            agents=[sudoku_coach],
            tasks=[task],
            verbose=False
        )

        result = crew.kickoff()
        return str(result)

    except Exception:
        return "Look for a row, column, or box with only one possible missing number."


# -----------------------------
# Sudoku Logic
# -----------------------------
def is_valid(board, row, col, num):
    """Check if num can be placed at board[row][col]"""

    if num in board[row]:
        return False

    if num in board[:, col]:
        return False

    start_row, start_col = 3 * (row // 3), 3 * (col // 3)
    for i in range(start_row, start_row + 3):
        for j in range(start_col, start_col + 3):
            if board[i][j] == num:
                return False

    return True


def solve_board(board):
    """Backtracking solver to fill Sudoku board"""
    for row in range(9):
        for col in range(9):
            if board[row][col] == 0:
                nums = list(range(1, 10))
                random.shuffle(nums)
                for num in nums:
                    if is_valid(board, row, col, num):
                        board[row][col] = num
                        if solve_board(board):
                            return True
                        board[row][col] = 0
                return False
    return True


def generate_full_board():
    """Generate a complete valid Sudoku solution"""
    board = np.zeros((9, 9), dtype=int)
    solve_board(board)
    return board


def make_puzzle(solution, difficulty="medium"):
    """Remove cells from full solution based on difficulty"""
    puzzle = solution.copy()

    if difficulty == "easy":
        remove_count = 45
    elif difficulty == "medium":
        remove_count = 52
    else:  # hard
        remove_count = 56

    cells = [(r, c) for r in range(9) for c in range(9)]
    random.shuffle(cells)

    for i in range(remove_count):
        r, c = cells[i]
        puzzle[r][c] = 0

    return puzzle


def generate_sudoku(difficulty):
    """Generate a brand-new valid Sudoku puzzle every time."""
    solution = generate_full_board()
    puzzle = make_puzzle(solution, difficulty)
    return puzzle, solution


def draw_sudoku(grid, title="Sudoku Puzzle"):
    """Draw Sudoku grid using matplotlib"""
    fig, ax = plt.subplots(figsize=(6, 6))

    for i in range(10):
        lw = 3 if i % 3 == 0 else 1
        ax.plot([0, 9], [i, i], color='black', linewidth=lw)
        ax.plot([i, i], [0, 9], color='black', linewidth=lw)

    for i in range(9):
        for j in range(9):
            value = grid[i, j]
            if value != 0:
                ax.text(
                    j + 0.5, i + 0.5, str(value),
                    ha='center', va='center',
                    fontsize=18, fontweight='bold'
                )

    ax.set_xlim(0, 9)
    ax.set_ylim(9, 0)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=18, fontweight='bold')

    st.pyplot(fig)
    plt.close(fig)


def clear_input_widgets():
    """Safely clear all editable widget keys so Streamlit recreates them next run."""
    for i in range(9):
        for j in range(9):
            key = f"input_{i}_{j}"
            if key in st.session_state:
                del st.session_state[key]


def sync_inputs_to_grid():
    """Read all text_input values into user_grid safely."""
    for i in range(9):
        for j in range(9):
            if st.session_state.puzzle[i, j] != 0:
                continue  # locked cells

            key = f"input_{i}_{j}"
            val = st.session_state.get(key, "")

            if str(val).isdigit() and 1 <= int(val) <= 9:
                st.session_state.user_grid[i, j] = int(val)
            else:
                st.session_state.user_grid[i, j] = 0


def preload_inputs_from_grid():
    """Prepare input widget values BEFORE widgets are rendered."""
    for i in range(9):
        for j in range(9):
            if st.session_state.puzzle[i, j] == 0:
                key = f"input_{i}_{j}"
                value = st.session_state.user_grid[i, j]
                st.session_state[key] = "" if value == 0 else str(value)


def render_editable_sudoku():
    """Render editable Sudoku input grid"""
    st.markdown("### ✍️ Fill the Puzzle")

    preload_inputs_from_grid()
    moves_count = 0

    for i in range(9):
        cols = st.columns(9)

        for j in range(9):
            original_value = st.session_state.puzzle[i, j]
            current_value = st.session_state.user_grid[i, j]

            with cols[j]:
                if original_value != 0:
                    st.markdown(
                        f"""
                        <div style="
                            text-align:center;
                            padding:10px;
                            border:2px solid black;
                            background-color:#4ECDC4;
                            color:black;
                            font-size:22px;
                            font-weight:bold;
                            border-radius:6px;
                            margin:2px;
                            min-height:50px;
                        ">
                            {original_value}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    key = f"input_{i}_{j}"
                    old_val = current_value

                    user_input = st.text_input(
                        label=f"cell_{i}_{j}",
                        value="" if current_value == 0 else str(current_value),
                        max_chars=1,
                        key=key,
                        label_visibility="collapsed"
                    )

                    if user_input.isdigit() and 1 <= int(user_input) <= 9:
                        st.session_state.user_grid[i, j] = int(user_input)
                    else:
                        st.session_state.user_grid[i, j] = 0

                    if old_val != st.session_state.user_grid[i, j]:
                        moves_count += 1

    st.session_state.moves = list(range(moves_count))


def reset_user_grid():
    st.session_state.user_grid = st.session_state.puzzle.copy()
    clear_input_widgets()


def start_new_game(difficulty):
    """Create a fresh puzzle automatically"""
    puzzle, solution = generate_sudoku(difficulty)
    st.session_state.puzzle = puzzle
    st.session_state.solution = solution
    st.session_state.user_grid = puzzle.copy()
    st.session_state.game_start = time.time()
    st.session_state.moves = []
    st.session_state.current_difficulty = difficulty
    st.session_state.ai_hint = ""
    clear_input_widgets()


def main():
    st.markdown("""
    <div style='text-align: center;'>
        <h1 style='color: #2FA26C; font-size: 4rem;'>🎮 Sudoku Master</h1>
        <br><br>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    st.sidebar.markdown("## 🎯 Game Settings")
    difficulty = st.sidebar.selectbox(
        "Difficulty Level",
        ["easy", "medium", "hard"],
        index=1
    )

    timer_mode = st.sidebar.checkbox("⏱️ Timer", value=True)
    hints_enabled = st.sidebar.checkbox("💡 Hints Available", value=True)

    if "current_difficulty" not in st.session_state:
        st.session_state.current_difficulty = difficulty

    if (
        "puzzle" not in st.session_state
        or st.session_state.current_difficulty != difficulty
    ):
        start_new_game(difficulty)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 🧩 Your Sudoku Puzzle")

        if timer_mode:
            elapsed = int(time.time() - st.session_state.game_start)
            st.markdown(f"**⏱️ Time: {elapsed // 60:02d}:{elapsed % 60:02d}**")

        draw_sudoku(st.session_state.user_grid, "Current Puzzle")
        render_editable_sudoku()

        st.markdown("---")

        col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)

        with col_btn1:
            if st.button("🔄 New Puzzle", use_container_width=True):
                start_new_game(difficulty)
                st.rerun()

        with col_btn2:
            if st.button("♻️ Reset", use_container_width=True):
                reset_user_grid()
                st.rerun()

        with col_btn3:
            if st.button("✅ Check Solution", use_container_width=True):
                sync_inputs_to_grid()
                if np.array_equal(st.session_state.user_grid, st.session_state.solution):
                    st.success("🎉 Congratulations! You solved it perfectly!")
                else:
                    st.error("❌ Not quite right. Keep trying!")

        with col_btn4:
            if st.button("💡 Hint", use_container_width=True) and hints_enabled:
                sync_inputs_to_grid()

                empty_cells = np.where(st.session_state.user_grid == 0)
                if len(empty_cells[0]) > 0:
                    idx = np.random.randint(0, len(empty_cells[0]))
                    i, j = empty_cells[0][idx], empty_cells[1][idx]
                    st.session_state.user_grid[i, j] = st.session_state.solution[i, j]
                    clear_input_widgets()
                    st.rerun()
                else:
                    st.info("No empty cells left!")

    

    with col2:
        st.markdown("### 📊 Game Stats")
        st.metric("Moves Made", int(np.count_nonzero(st.session_state.user_grid - st.session_state.puzzle)))
        st.metric("Difficulty", difficulty.title())

        if timer_mode:
            elapsed = int(time.time() - st.session_state.game_start)
            st.metric("Elapsed Time", f"{elapsed // 60:02d}:{elapsed % 60:02d}")

        filled_cells = np.count_nonzero(st.session_state.user_grid)
        st.metric("Filled Cells", f"{filled_cells}/81")

        st.markdown("### 🎮 Controls")
        st.info("""
        **Type numbers** in empty cells below the board  
        **🔄 New Puzzle** - Generate fresh puzzles  
        **♻️ Reset** - Clear your progress  
        **✅ Check** - Validate solution  
        **💡 Hint** - Fill one correct cell  
        **👑 Show Solution** - Reveal solved board
        """)

        if st.button("👑 Show Solution", use_container_width=True):
            draw_sudoku(st.session_state.solution, "Complete Solution")


if __name__ == "__main__":
    main()
