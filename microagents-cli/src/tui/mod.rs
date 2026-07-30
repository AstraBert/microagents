//! Minimal, aesthetically-pleasing TUI for chatting with a [`MicroAgent`].
//!
//! The TUI is intentionally backend-agnostic: it does not own the agent, but
//! accepts a closure that knows how to start a new run for a given prompt and
//! (optional) session id. This makes it easy to plug in any concrete
//! [`MicroAgent<Ctx>`](microagents_core::agent::MicroAgent) since `Agent::run`
//! consumes the agent — the closure can rebuild it on every turn while keeping
//! continuity through the session id.
//!
//! ```ignore
//! use microagents_cli::tui;
//! tui::run(|prompt, session_id| async move {
//!     build_agent().run(prompt, session_id).await
//! }).await?;
//! ```
use std::{
    cmp::Reverse,
    collections::HashSet,
    future::Future,
    io::{self, Stdout},
    path::{Path, PathBuf},
    time::Duration,
};

use futures_util::StreamExt;
use ignore::WalkBuilder;
use microagents_core::types::{AgentError, RunStream};
use microagents_events::{
    AgentEventAny, AssistantMessagePart, DeltaType, TaskStatus, ToolCallPart, types::ToolResult,
};
use ratatui::{
    Frame, Terminal,
    crossterm::{
        event::{self, Event, KeyCode, KeyEventKind, KeyModifiers},
        execute,
        terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
    },
    layout::{Constraint, Direction, Layout, Rect},
    prelude::CrosstermBackend,
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{
        Block, BorderType, Borders, Padding, Paragraph, Scrollbar, ScrollbarOrientation,
        ScrollbarState, Wrap,
    },
};
use tokio::sync::mpsc;
use unicode_width::UnicodeWidthStr;

/// Characters that delimit tokens for @-mention detection.
const PATH_DELIMITERS: &[char] = &[' ', '\t', '"', '\'', '='];

/// Check if a character is a path delimiter.
fn is_path_delimiter(c: char) -> bool {
    PATH_DELIMITERS.contains(&c)
}

/// Find the start of the current @-mention token at the given cursor position.
/// Returns `Some((start_byte_offset, prefix_including_at))` if the cursor is
/// inside or immediately after an `@` token.
fn find_at_prefix(input: &str, cursor: usize) -> Option<(usize, String)> {
    // Find the last delimiter before cursor
    let before = &input[..cursor];
    let last_delim = before
        .rfind(is_path_delimiter)
        .map(|i| i as isize)
        .unwrap_or(-1);
    let start = (last_delim + 1) as usize;
    let token = &input[start..cursor];
    if token.starts_with('@') {
        Some((start, token.to_string()))
    } else {
        None
    }
}

/// Score a file path against a query (higher = better).
fn score_entry(name: &str, path: &str, query: &str, is_dir: bool) -> u32 {
    let ln = name.to_lowercase();
    let lq = query.to_lowercase();
    let mut score = 0u32;
    if ln == lq {
        score = 100;
    } else if ln.starts_with(&lq) {
        score = 80;
    } else if ln.contains(&lq) {
        score = 50;
    } else if path.to_lowercase().contains(&lq) {
        score = 30;
    }
    if is_dir && score > 0 {
        score += 10;
    }
    score
}

/// Collect fuzzy file suggestions from cwd, respecting .gitignore.
fn collect_suggestions(cwd: &Path, query: &str) -> Vec<Suggestion> {
    let mut out = Vec::new();
    let mut seen = HashSet::new();

    // Determine search scope: if query contains '/', scope to that directory.
    let (base_dir, file_query) = if let Some(slash) = query.rfind('/') {
        let dir_part = &query[..=slash];
        let file_part = &query[slash + 1..];
        let resolved = if dir_part.starts_with("~/") {
            dirs::home_dir()
                .map(|h| h.join(&dir_part[2..]))
                .unwrap_or_else(|| cwd.join(&dir_part[1..]))
        } else if dir_part.starts_with('/') {
            PathBuf::from(dir_part)
        } else {
            cwd.join(dir_part)
        };
        (resolved, file_part)
    } else {
        (cwd.to_path_buf(), query)
    };

    let walker = WalkBuilder::new(&base_dir)
        .max_depth(Some(6))
        .hidden(false)
        .add_custom_ignore_filename(".microagentsignore")
        .follow_links(false)
        .build();

    for entry in walker {
        let Ok(entry) = entry else { continue };
        let is_dir = entry.file_type().unwrap().is_dir();
        let name = entry.file_name().to_string_lossy().into_owned();

        // Skip root
        if entry.depth() == 0 {
            continue;
        }

        let rel_path = entry.path().strip_prefix(&base_dir).unwrap_or(entry.path());
        let display_path = if query.contains('/') {
            // Preserve the prefix the user typed
            let prefix_dir = &query[..query.rfind('/').map(|i| i + 1).unwrap_or(0)];
            if prefix_dir.is_empty() {
                format!("{}", rel_path.display())
            } else {
                format!("{}/{}", prefix_dir, rel_path.display())
            }
        } else {
            format!("{}", rel_path.display())
        };

        let score = if file_query.is_empty() {
            1
        } else {
            score_entry(&name, &display_path, file_query, is_dir)
        };

        if score == 0 {
            continue;
        }

        let key = entry.path().to_path_buf();
        if !seen.insert(key) {
            continue;
        }

        out.push(Suggestion {
            name: name.clone(),
            path: display_path,
            is_dir,
            score,
        });
    }

    // Sort by score desc, then dirs first, then name
    out.sort_by_key(|s| (Reverse(s.score), s.is_dir, s.name.clone()));
    out.truncate(50);
    out
}

/// Build the completion value to insert (preserves @, adds quotes if needed).
fn build_completion_value(path: &str, is_dir: bool, is_quoted: bool) -> String {
    let needs_quotes = is_quoted || path.contains(' ');
    let p = if is_dir && !path.ends_with('/') {
        format!("{}/", path)
    } else {
        path.to_string()
    };
    if needs_quotes {
        format!("@\"{}\"", p)
    } else {
        format!("@{}", p)
    }
}

/// Palette tuned for both light and dark terminals.
mod theme {
    use ratatui::style::Color;

    pub const ACCENT: Color = Color::Rgb(137, 180, 250); // soft blue
    pub const ACCENT_SOFT: Color = Color::Rgb(116, 199, 236);
    pub const USER: Color = Color::Rgb(166, 227, 161); // mint
    pub const ASSISTANT: Color = Color::Rgb(205, 214, 244); // off-white
    pub const THINKING: Color = Color::Rgb(147, 153, 178); // muted
    pub const TOOL: Color = Color::Rgb(249, 226, 175); // soft yellow
    pub const TOOL_OK: Color = Color::Rgb(166, 227, 161);
    pub const TOOL_ERR: Color = Color::Rgb(243, 139, 168);
    pub const SKILL: Color = Color::Rgb(203, 166, 247); // lavender
    pub const TASK: Color = Color::Rgb(0, 236, 128); // light green
    pub const DIM: Color = Color::Rgb(108, 112, 134);
    pub const ERROR: Color = Color::Rgb(243, 139, 168);
}

/// A single rendered line in the chat transcript.
#[derive(Debug, Clone)]
enum Msg {
    User(String),
    Assistant(String),
    Thinking(String),
    ToolCall { name: String, input: String },
    ToolResult(ToolResult),
    Skill(String),
    Task { name: String, status: TaskStatus },
    Session(String),
    Error(String),
}

#[derive(Debug)]
enum UiEvent {
    Agent(AgentEventAny),
    AgentError(AgentError),
    /// The current run has finished — the input is unlocked.
    RunFinished,
}

/// Bounds for the dynamic-height input box (in *visual* rows, content only).
const INPUT_MIN_ROWS: u16 = 1;
const INPUT_MAX_ROWS: u16 = 8;

/// Max visible items in the @-file suggestion popup.
const SUGGEST_MAX_VISIBLE: usize = 8;

/// A single file/directory suggestion.
#[derive(Debug, Clone)]
struct Suggestion {
    name: String,
    path: String,
    is_dir: bool,
    /// Match score (higher = better).
    score: u32,
}

/// State for the @-file suggestion popup.
#[derive(Debug, Default)]
struct SuggestState {
    /// Byte offset in `input` where the prefix starts.
    start: usize,
    /// Filtered + scored suggestions.
    items: Vec<Suggestion>,
    /// Selected index.
    selected: usize,
    /// Whether the popup is active.
    active: bool,
    /// Whether the prefix was quoted (e.g. `@"src`).
    is_quoted: bool,
}

struct App {
    /// Raw input buffer. May contain '\n' for multi-line prompts.
    input: String,
    /// Byte offset of the caret inside `input`.
    cursor: usize,
    transcript: Vec<Msg>,
    session_id: Option<String>,
    busy: bool,
    /// Current scroll offset in *visual* (post-wrap) rows from the top of the transcript.
    scroll: u16,
    auto_scroll: bool,
    quit: bool,
    /// Visual height of the transcript content after width-wrapping (last frame).
    last_content_height: u16,
    /// Visual height of the transcript viewport (last frame).
    last_viewport_height: u16,
    /// @-file suggestion popup state.
    suggest: SuggestState,
    /// Working directory for file suggestions.
    cwd: PathBuf,
}

impl App {
    /// Refresh suggestions based on current input + cursor.
    fn refresh_suggestions(&mut self) {
        if let Some((start, prefix)) = find_at_prefix(&self.input, self.cursor) {
            let raw = &prefix[1..]; // strip '@'
            let is_quoted = raw.starts_with('"');
            let query = if is_quoted { &raw[1..] } else { raw };
            let items = collect_suggestions(&self.cwd, query);
            self.suggest = SuggestState {
                start,
                items,
                selected: 0,
                active: true,
                is_quoted,
            };
        } else {
            self.suggest.active = false;
        }
    }

    /// Accept the currently selected suggestion into the input buffer.
    fn accept_suggestion(&mut self) {
        if !self.suggest.active || self.suggest.items.is_empty() {
            return;
        }
        let sel = &self.suggest.items[self.suggest.selected];
        let value = build_completion_value(&sel.path, sel.is_dir, self.suggest.is_quoted);
        let before = &self.input.clone()[..self.suggest.start];
        let after = &self.input[self.cursor..];
        // For directories, don't add trailing space so user can keep typing
        let suffix = if sel.is_dir { "" } else { " " };
        self.input = format!("{}{}{}{}", before, value, suffix, after);
        self.cursor = before.len() + value.len() + suffix.len();
        self.suggest.active = false;
    }

    /// Move selection up.
    fn suggest_prev(&mut self) {
        if self.suggest.selected > 0 {
            self.suggest.selected -= 1;
        }
    }

    /// Move selection down.
    fn suggest_next(&mut self) {
        if self.suggest.selected + 1 < self.suggest.items.len() {
            self.suggest.selected += 1;
        }
    }
}

impl App {
    fn new(session_id: Option<String>) -> Self {
        Self {
            input: String::new(),
            cursor: 0,
            transcript: Vec::new(),
            session_id,
            busy: false,
            scroll: 0,
            auto_scroll: true,
            quit: false,
            last_content_height: 0,
            last_viewport_height: 0,
            suggest: SuggestState::default(),
            cwd: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
        }
    }

    fn push(&mut self, block: Msg) {
        self.transcript.push(block);
        if self.auto_scroll {
            self.scroll = u16::MAX; // clamp in draw
        }
    }

    /// Append streaming text to the current assistant block, creating one if needed.
    fn append_assistant_delta(&mut self, delta: &str, thinking: bool) {
        if delta.is_empty() {
            return;
        }
        let target_is_match = match self.transcript.last() {
            Some(Msg::Assistant(_)) if !thinking => true,
            Some(Msg::Thinking(_)) if thinking => true,
            _ => false,
        };
        if !target_is_match {
            self.transcript.push(if thinking {
                Msg::Thinking(String::new())
            } else {
                Msg::Assistant(String::new())
            });
        }
        match self.transcript.last_mut().unwrap() {
            Msg::Assistant(s) | Msg::Thinking(s) => s.push_str(delta),
            _ => unreachable!(),
        }
        if self.auto_scroll {
            self.scroll = u16::MAX;
        }
    }

    fn apply(&mut self, ev: UiEvent) {
        match ev {
            UiEvent::Agent(e) => self.apply_agent_event(e, false),
            UiEvent::AgentError(e) => self.push(Msg::Error(e.to_string())),
            UiEvent::RunFinished => self.busy = false,
        }
    }

    pub fn apply_agent_event(&mut self, ev: AgentEventAny, is_replay: bool) {
        match ev {
            AgentEventAny::SessionInit(s) => {
                self.session_id = Some(s.session_id.clone());
                let kind = match s.init_type {
                    microagents_events::SessionInitType::Start => "started",
                    microagents_events::SessionInitType::Resume => "resumed",
                    _ => unreachable!("SessionInitType should not reach this branch"),
                };
                self.push(Msg::Session(format!(
                    "session {} • {} • {}/{}",
                    kind, s.session_id, s.provider, s.model
                )));
            }
            AgentEventAny::SessionStop(s) => {
                if let Some(err) = s.error {
                    self.push(Msg::Error(err));
                }
                self.push(Msg::Session(format!(
                    "session stopped • {} • {:.2}s • {:?} input toks ({:?} cached) • {:?} output toks",
                    if s.success { "ok" } else { "failed" },
                    s.usage.latency as f64 / 1000_f64,
                    s.usage.input_tokens,
                    s.usage.cache_read_tokens,
                    s.usage.output_tokens,
                )));
            }
            AgentEventAny::UserPromptSubmit(m) => {
                if is_replay {
                    self.push(Msg::User(m.prompt));
                }
            }
            AgentEventAny::StreamDelta(d) => {
                let thinking = matches!(d.delta_type, DeltaType::Thinking);
                self.append_assistant_delta(&d.delta, thinking);
            }
            AgentEventAny::ToolCall(t) => {
                let input = serde_json::to_string(&t.input).unwrap_or_else(|_| "{}".into());
                self.push(Msg::ToolCall {
                    name: t.name,
                    input,
                });
            }
            AgentEventAny::ToolResult(r) => self.push(Msg::ToolResult(r.result)),
            AgentEventAny::SkillLoad(s) => self.push(Msg::Skill(s.skill_name)),
            AgentEventAny::AssistantResponse(r) => {
                // Streamed text is already rendered via deltas. The core does
                // not currently emit `tool.call` events, so derive them from
                // the final assistant response's `tool_calls` field.
                let mut tool_calls: Option<Vec<ToolCallPart>> = None;
                for part in r.content {
                    match part {
                        AssistantMessagePart::ToolCall(t) => {
                            tool_calls.get_or_insert_with(Vec::new).push(t)
                        }
                        _ => continue,
                    }
                }
                if let Some(calls) = tool_calls {
                    for c in calls {
                        let input = serde_json::from_str::<serde_json::Value>(&c.arguments)
                            .ok()
                            .and_then(|v| serde_json::to_string(&v).ok())
                            .unwrap_or(c.arguments);
                        self.push(Msg::ToolCall {
                            name: c.name,
                            input,
                        });
                    }
                }
            }
            AgentEventAny::Task(t) => self.push(Msg::Task {
                name: t.task_name,
                status: t.task_status,
            }),
            // no-op, these are already handled by other branches
            AgentEventAny::ToolAnyCall(_) => {}
            _ => unreachable!("AgentEventAny should not reach this branch"),
        }
    }
}

/// Start an interactive TUI session, optionally resuming a previous one.
///
/// When `session_id` is `Some(id)`, the first call to `start_run` will receive
/// that id, allowing the agent to resume the conversation from storage. When
/// `None`, behaves identically to [`run`].
///
/// `load_history` is called once at startup when `session_id` is `Some`. It
/// should return the historical events for that session so the TUI can
/// pre-populate the transcript before any user input.
pub async fn run_with_session<F, Fut, H, Hfut>(
    session_id: Option<String>,
    mut start_run: F,
    load_history: H,
) -> io::Result<()>
where
    F: FnMut(String, Option<String>) -> Fut + Send + 'static,
    Fut: Future<Output = Result<RunStream, AgentError>> + Send + 'static,
    H: FnOnce(String) -> Hfut + Send + 'static,
    Hfut: Future<Output = Result<Vec<AgentEventAny>, AgentError>> + Send + 'static,
{
    let mut terminal = setup_terminal()?;
    let history = if let Some(ref sid) = session_id {
        match load_history(sid.clone()).await {
            Ok(events) => events,
            Err(e) => {
                restore_terminal(&mut terminal)?;
                return Err(io::Error::other(e.to_string()));
            }
        }
    } else {
        vec![]
    };
    let res = event_loop(&mut terminal, &mut start_run, session_id, history).await;
    restore_terminal(&mut terminal)?;
    res
}

fn setup_terminal() -> io::Result<Terminal<CrosstermBackend<Stdout>>> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    Terminal::new(CrosstermBackend::new(stdout))
}

fn restore_terminal(terminal: &mut Terminal<CrosstermBackend<Stdout>>) -> io::Result<()> {
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    Ok(())
}

async fn event_loop<F, Fut>(
    terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    start_run: &mut F,
    session_id: Option<String>,
    history: Vec<AgentEventAny>,
) -> io::Result<()>
where
    F: FnMut(String, Option<String>) -> Fut,
    Fut: Future<Output = Result<RunStream, AgentError>> + Send + 'static,
{
    let mut app = App::new(session_id);
    for ev in history {
        app.apply_agent_event(ev, true);
    }
    let (ui_tx, mut ui_rx) = mpsc::unbounded_channel::<UiEvent>();
    let mut tick = tokio::time::interval(Duration::from_millis(80));

    while !app.quit {
        terminal.draw(|f| draw(f, &mut app))?;

        tokio::select! {
            biased;
            Some(ev) = ui_rx.recv() => {
                app.apply(ev);
                // Drain anything else that's immediately available to keep streams smooth.
                while let Ok(more) = ui_rx.try_recv() {
                    app.apply(more);
                }
            }
            _ = tick.tick() => {
                while event::poll(Duration::from_millis(0))? {
                    if let Event::Key(key) = event::read()? {
                        if key.kind != KeyEventKind::Press { continue; }
                        handle_key(key, &mut app, start_run, &ui_tx).await;
                    }
                }
            }
        }
    }
    Ok(())
}

async fn handle_key<F, Fut>(
    key: event::KeyEvent,
    app: &mut App,
    start_run: &mut F,
    ui_tx: &mpsc::UnboundedSender<UiEvent>,
) where
    F: FnMut(String, Option<String>) -> Fut,
    Fut: Future<Output = Result<RunStream, AgentError>> + Send + 'static,
{
    let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
    match key.code {
        KeyCode::Char('c') if ctrl => app.quit = true,
        KeyCode::Char('d') if ctrl && app.input.is_empty() => app.quit = true,
        KeyCode::Esc => {
            if app.suggest.active {
                app.suggest.active = false;
                return;
            }
            app.quit = true;
        }
        KeyCode::Enter if !app.busy => {
            if app.suggest.active {
                app.accept_suggestion();
                return;
            }
            if key.modifiers.contains(KeyModifiers::SHIFT) {
                app.input.insert(app.cursor, '\n');
                app.cursor += 1;
                return;
            }
            let prompt = app.input.trim().to_string();
            if prompt.is_empty() {
                return;
            }
            if matches!(prompt.as_str(), "/exit" | "/quit") {
                app.quit = true;
                return;
            }
            app.input.clear();
            app.cursor = 0;
            app.push(Msg::User(prompt.clone()));
            app.busy = true;
            app.auto_scroll = true;

            let fut = start_run(prompt, app.session_id.clone());
            let tx = ui_tx.clone();
            tokio::spawn(async move {
                match fut.await {
                    Ok(mut stream) => {
                        while let Some(item) = stream.next().await {
                            match item {
                                Ok(e) => {
                                    if tx.send(UiEvent::Agent(e)).is_err() {
                                        return;
                                    }
                                }
                                Err(e) => {
                                    let _ = tx.send(UiEvent::AgentError(e));
                                }
                            }
                        }
                    }
                    Err(e) => {
                        let _ = tx.send(UiEvent::AgentError(e));
                    }
                }
                let _ = tx.send(UiEvent::RunFinished);
            });
        }
        KeyCode::Char(c) if !ctrl => {
            app.input.insert(app.cursor, c);
            app.cursor += c.len_utf8();
            if c == '@' || (app.suggest.active && c != ' ') {
                app.refresh_suggestions();
            } else if c == ' ' {
                app.suggest.active = false;
            }
        }
        KeyCode::Tab if !app.busy => {
            if !app.suggest.active {
                // Try to trigger suggestions if cursor is after @
                app.refresh_suggestions();
            }
            if app.suggest.active {
                app.accept_suggestion();
            }
        }
        KeyCode::Up if app.suggest.active => {
            app.suggest_prev();
        }
        KeyCode::Down if app.suggest.active => {
            app.suggest_next();
        }
        KeyCode::Backspace => {
            if app.cursor > 0 {
                let mut new_cursor = app.cursor - 1;
                while !app.input.is_char_boundary(new_cursor) && new_cursor > 0 {
                    new_cursor -= 1;
                }
                app.input.replace_range(new_cursor..app.cursor, "");
                app.cursor = new_cursor;
                app.refresh_suggestions();
            }
        }
        KeyCode::Left => {
            if app.cursor > 0 {
                let mut nc = app.cursor - 1;
                while !app.input.is_char_boundary(nc) && nc > 0 {
                    nc -= 1;
                }
                app.cursor = nc;
            }
        }
        KeyCode::Right => {
            if app.cursor < app.input.len() {
                let mut nc = app.cursor + 1;
                while nc < app.input.len() && !app.input.is_char_boundary(nc) {
                    nc += 1;
                }
                app.cursor = nc;
            }
        }
        KeyCode::Home => app.cursor = 0,
        KeyCode::End => app.cursor = app.input.len(),
        KeyCode::PageUp => {
            app.auto_scroll = false;
            app.scroll = app.scroll.saturating_sub(app.last_viewport_height.max(1));
        }
        KeyCode::PageDown => {
            let max = app
                .last_content_height
                .saturating_sub(app.last_viewport_height);
            app.scroll = app.scroll.saturating_add(app.last_viewport_height).min(max);
            app.auto_scroll = app.scroll >= max;
        }
        _ => {}
    }
}

fn draw(f: &mut Frame, app: &mut App) {
    let input_height = input_visual_height(&app.input, f.area().width);
    let suggest_height = if app.suggest.active {
        (app.suggest.items.len().min(SUGGEST_MAX_VISIBLE) as u16 + 2).min(f.area().height / 2)
    } else {
        0
    };

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),              // header
            Constraint::Min(1),                 // chat
            Constraint::Length(suggest_height), // suggestion popup
            Constraint::Length(input_height),   // input (dynamic)
            Constraint::Length(1),              // hint
        ])
        .split(f.area());

    draw_header(f, chunks[0], app);
    draw_transcript(f, chunks[1], app);
    if app.suggest.active {
        draw_suggestions(f, chunks[2], app);
    }
    draw_input(f, chunks[3], app);
    draw_hint(f, chunks[4], app);
}

fn draw_header(f: &mut Frame, area: Rect, app: &App) {
    let session = app.session_id.as_deref().unwrap_or("—");
    let status = if app.busy { "● thinking" } else { "○ idle" };
    let status_color = if app.busy {
        theme::ACCENT_SOFT
    } else {
        theme::DIM
    };

    let title = Line::from(vec![
        Span::styled("✦ ", Style::default().fg(theme::ACCENT)),
        Span::styled(
            "microagents",
            Style::default()
                .fg(theme::ACCENT)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw("  "),
        Span::styled(
            format!("session {}", session),
            Style::default().fg(theme::DIM),
        ),
        Span::raw("  "),
        Span::styled(status, Style::default().fg(status_color)),
    ]);

    let block = Block::default()
        .borders(Borders::BOTTOM)
        .border_type(BorderType::Plain)
        .border_style(Style::default().fg(theme::DIM));
    let p = Paragraph::new(title).block(block).left_aligned();
    f.render_widget(p, area);
}

fn block_lines(block: &Msg) -> Vec<Line<'static>> {
    match block {
        Msg::User(text) => prefixed("  you", theme::USER, text, theme::ASSISTANT),
        Msg::Assistant(text) => prefixed("  agent", theme::ACCENT, text, theme::ASSISTANT),
        Msg::Thinking(text) => {
            let lines = wrap_to_lines(text);
            let mut out = Vec::with_capacity(lines.len() + 1);
            out.push(Line::from(Span::styled(
                "  · thinking",
                Style::default()
                    .fg(theme::THINKING)
                    .add_modifier(Modifier::ITALIC),
            )));
            for l in lines {
                out.push(Line::from(Span::styled(
                    format!("    {}", l),
                    Style::default()
                        .fg(theme::THINKING)
                        .add_modifier(Modifier::ITALIC),
                )));
            }
            out
        }
        Msg::ToolCall { name, input } => {
            let preview = if input.len() > 400 {
                format!("{}…", &input[..400])
            } else {
                input.clone()
            };
            vec![
                Line::from(Span::styled(
                    "  tool",
                    Style::default()
                        .fg(theme::TOOL)
                        .add_modifier(Modifier::BOLD),
                )),
                Line::from(vec![
                    Span::styled("    ↪ ", Style::default().fg(theme::TOOL)),
                    Span::styled(
                        name.clone(),
                        Style::default()
                            .fg(theme::TOOL)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::raw("  "),
                    Span::styled(preview, Style::default().fg(theme::DIM)),
                ]),
            ]
        }
        Msg::ToolResult(r) => match r {
            ToolResult::Ok(s) => {
                let actual_res = if s.len() > 200 {
                    s[..s.floor_char_boundary(200)].to_string() + "..."
                } else {
                    s.to_string()
                };
                prefixed_inline("  ✓ ", theme::TOOL_OK, &actual_res, theme::DIM)
            }
            ToolResult::Err(s) => {
                let actual_res = if s.len() > 200 {
                    s[..s.len().min(200)].to_string() + "..."
                } else {
                    s.to_string()
                };
                prefixed_inline("  ✗ ", theme::TOOL_ERR, &actual_res, theme::TOOL_ERR)
            }
            _ => unreachable!("ToolResult should not reach this branch"),
        },
        Msg::Skill(name) => vec![Line::from(vec![
            Span::styled("  ✧ ", Style::default().fg(theme::SKILL)),
            Span::styled(
                format!("skill loaded: {}", name),
                Style::default().fg(theme::SKILL),
            ),
        ])],
        Msg::Task { name, status } => vec![Line::from(vec![
            Span::styled("  ✧ ", Style::default().fg(theme::TASK)),
            Span::styled(
                format!("task: {}   status: {}", name, status),
                Style::default().fg(theme::TASK),
            ),
        ])],
        Msg::Session(msg) => vec![Line::from(Span::styled(
            format!("  ── {}", msg),
            Style::default()
                .fg(theme::DIM)
                .add_modifier(Modifier::ITALIC),
        ))],
        Msg::Error(msg) => vec![Line::from(vec![
            Span::styled("  ! ", Style::default().fg(theme::ERROR)),
            Span::styled(
                msg.clone(),
                Style::default()
                    .fg(theme::ERROR)
                    .add_modifier(Modifier::BOLD),
            ),
        ])],
    }
}

fn prefixed(label: &str, label_color: Color, text: &str, body_color: Color) -> Vec<Line<'static>> {
    let mut out = vec![Line::from(Span::styled(
        label.to_string(),
        Style::default()
            .fg(label_color)
            .add_modifier(Modifier::BOLD),
    ))];
    for l in wrap_to_lines(text) {
        out.push(Line::from(Span::styled(
            format!("    {}", l),
            Style::default().fg(body_color),
        )));
    }
    out
}

fn prefixed_inline(
    prefix: &str,
    prefix_color: Color,
    text: &str,
    body_color: Color,
) -> Vec<Line<'static>> {
    let lines = wrap_to_lines(text);
    if lines.is_empty() {
        return vec![Line::from(Span::styled(
            prefix.to_string(),
            Style::default().fg(prefix_color),
        ))];
    }
    let mut out = Vec::with_capacity(lines.len());
    for (i, l) in lines.into_iter().enumerate() {
        if i == 0 {
            out.push(Line::from(vec![
                Span::styled(
                    prefix.to_string(),
                    Style::default()
                        .fg(prefix_color)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(l, Style::default().fg(body_color)),
            ]));
        } else {
            out.push(Line::from(Span::styled(
                format!("    {}", l),
                Style::default().fg(body_color),
            )));
        }
    }
    out
}

/// Break text on existing newlines; ratatui's wrap handles line-width wrapping.
fn wrap_to_lines(text: &str) -> Vec<String> {
    if text.is_empty() {
        return Vec::new();
    }
    text.split('\n').map(|s| s.to_string()).collect()
}

fn draw_transcript(f: &mut Frame, area: Rect, app: &mut App) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(theme::DIM))
        .padding(Padding::new(1, 1, 0, 0));

    let inner = block.inner(area);
    f.render_widget(block, area);

    let mut lines: Vec<Line<'static>> = Vec::new();
    for (i, b) in app.transcript.iter().enumerate() {
        if i > 0 {
            lines.push(Line::from(""));
        }
        lines.extend(block_lines(b));
    }
    if lines.is_empty() {
        lines.push(Line::from(Span::styled(
            "  Ask anything to get started.",
            Style::default().fg(theme::DIM).add_modifier(Modifier::DIM),
        )));
    }

    // Compute wrapped height so auto-scroll actually reaches the bottom.
    let wrap_width = inner.width.max(1);
    let total = wrapped_line_count(&lines, wrap_width);
    let viewport = inner.height;
    app.last_content_height = total;
    app.last_viewport_height = viewport;

    let max_scroll = total.saturating_sub(viewport);
    if app.auto_scroll || app.scroll > max_scroll {
        app.scroll = max_scroll;
    }

    let paragraph = Paragraph::new(lines)
        .wrap(Wrap { trim: false })
        .scroll((app.scroll, 0));
    f.render_widget(paragraph, inner);

    if total > viewport {
        let mut sb_state = ScrollbarState::new(total as usize).position(app.scroll as usize);
        let scrollbar = Scrollbar::new(ScrollbarOrientation::VerticalRight)
            .style(Style::default().fg(theme::DIM));
        f.render_stateful_widget(scrollbar, area, &mut sb_state);
    }
}

/// Approximate the number of visual rows `lines` will occupy when wrapped to `width`.
///
/// Mirrors ratatui's `Wrap { trim: false }` behavior: every logical line always
/// occupies at least one visual row (even when empty), plus extra rows for
/// content that overflows `width`.
fn wrapped_line_count(lines: &[Line<'_>], width: u16) -> u16 {
    let w = width.max(1);
    let mut count: u32 = 0;
    for line in lines {
        let line_width: u32 = line.spans.iter().map(|s| s.content.width() as u32).sum();
        // ceil(line_width / w), with a floor of 1 so empty lines still take a row.
        let rows = line_width.div_ceil(w as u32).max(1);
        count = count.saturating_add(rows);
    }
    count.min(u16::MAX as u32) as u16
}

fn input_visual_height(input: &str, area_width: u16) -> u16 {
    let usable = area_width.saturating_sub(4); // borders(2) + horizontal padding(2)
    let prefix_cols = 2; // "▎ "
    let wrap_width = usable.saturating_sub(prefix_cols).max(1);
    let mut rows: u16 = 1;
    for line in input.split('\n') {
        let line_len = line.chars().count() as u16;
        rows += line_len.saturating_sub(1) / wrap_width + 1;
    }
    rows.clamp(INPUT_MIN_ROWS, INPUT_MAX_ROWS) + 2 // +2 for borders
}

fn draw_input(f: &mut Frame, area: Rect, app: &App) {
    let border_color = if app.busy { theme::DIM } else { theme::ACCENT };
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(border_color))
        .padding(Padding::horizontal(1));

    let prompt_marker = Span::styled(
        "▎ ",
        Style::default()
            .fg(theme::ACCENT)
            .add_modifier(Modifier::BOLD),
    );

    let lines: Vec<Line> = if app.input.is_empty() && !app.busy {
        vec![Line::from(vec![
            prompt_marker,
            Span::styled(
                "Type a message and press Enter…",
                Style::default().fg(theme::DIM).add_modifier(Modifier::DIM),
            ),
        ])]
    } else if app.busy && app.input.is_empty() {
        vec![Line::from(vec![
            prompt_marker,
            Span::styled(
                "Waiting for the agent…",
                Style::default()
                    .fg(theme::DIM)
                    .add_modifier(Modifier::ITALIC),
            ),
        ])]
    } else {
        let mut out = Vec::new();
        for (i, line_text) in app.input.split('\n').enumerate() {
            if i == 0 {
                out.push(Line::from(vec![
                    prompt_marker.clone(),
                    Span::styled(line_text.to_string(), Style::default().fg(theme::ASSISTANT)),
                ]));
            } else {
                out.push(Line::from(vec![
                    Span::styled("   ", Style::default()),
                    Span::styled(line_text.to_string(), Style::default().fg(theme::ASSISTANT)),
                ]));
            }
        }
        out
    };

    let p = Paragraph::new(lines)
        .block(block.clone())
        .wrap(Wrap { trim: false });
    f.render_widget(p, area);

    // Cursor placement (only when accepting input).
    if !app.busy {
        let inner = block.inner(area);
        let prefix_cols = 2u16; // "▎ "
        let wrap_width = inner.width.saturating_sub(prefix_cols).max(1);

        let text_before_cursor = &app.input[..app.cursor];
        let line_index = text_before_cursor.matches('\n').count() as u16;
        let current_line_start = text_before_cursor.rfind('\n').map(|i| i + 1).unwrap_or(0);
        let current_line_text = &app.input[current_line_start..app.cursor];
        let col_in_line = current_line_text.chars().count() as u16;

        let cx = if line_index == 0 {
            inner.x + prefix_cols + col_in_line.min(wrap_width)
        } else {
            inner.x + col_in_line % wrap_width
        };
        let cy = inner.y + line_index + col_in_line / wrap_width;
        if cy < inner.y + inner.height {
            f.set_cursor_position((cx, cy));
        }
    }
}

fn draw_suggestions(f: &mut Frame, area: Rect, app: &App) {
    if app.suggest.items.is_empty() {
        return;
    }
    let visible = app.suggest.items.len().min(SUGGEST_MAX_VISIBLE);
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(theme::ACCENT))
        .padding(Padding::horizontal(1));

    let inner = block.inner(area);
    let max_name_width = inner.width.saturating_mul(2).saturating_div(5).max(8) as usize;

    let mut lines: Vec<Line> = Vec::with_capacity(visible);
    for (i, item) in app.suggest.items.iter().enumerate().take(visible) {
        let is_selected = i == app.suggest.selected;
        let name = if item.name.width() > max_name_width {
            format!(
                "{}…",
                &item.name[..item.name.floor_char_boundary(max_name_width - 1)]
            )
        } else {
            item.name.clone()
        };
        let icon = if item.is_dir { "📁 " } else { "📄 " };
        let name_span = Span::styled(
            format!("{}{}", icon, name),
            if is_selected {
                Style::default()
                    .fg(theme::ACCENT)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(theme::ASSISTANT)
            },
        );
        let path_span = Span::styled(
            format!("  {}", item.path),
            Style::default()
                .fg(theme::DIM)
                .add_modifier(Modifier::ITALIC),
        );
        lines.push(Line::from(vec![name_span, path_span]));
    }

    let p = Paragraph::new(lines).block(block);
    f.render_widget(p, area);
}

fn draw_hint(f: &mut Frame, area: Rect, app: &App) {
    let hint = if app.busy {
        "  streaming…  •  Ctrl+C quit"
    } else if app.suggest.active {
        "  ↑↓ navigate  •  Tab/Enter accept  •  Esc cancel"
    } else {
        "  Enter send  •  PgUp/PgDn scroll  •  /exit or Esc to quit"
    };
    let p = Paragraph::new(Line::from(Span::styled(
        hint,
        Style::default().fg(theme::DIM),
    )));
    f.render_widget(p, area);
}
