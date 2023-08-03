use std::io::{Read, Seek, SeekFrom, Write};

use bytemuck::Zeroable;
use marlinformat::PackedBoard;
use structopt::StructOpt;

use cozy_chess::{Color, Square};

#[derive(StructOpt)]
/// Report statistics about a dataset
pub struct Options {
    dataset: std::path::PathBuf,
}

pub fn run(options: Options) -> Result<(), Box<dyn std::error::Error>> {
    let mut dataset = std::fs::File::open(options.dataset)?;
    let size_bytes = dataset.seek(SeekFrom::End(0))?;
    dataset.seek(SeekFrom::Start(0))?;
    let mut count = size_bytes / std::mem::size_of::<PackedBoard>() as u64;
    let positions = count;
    let mut reader = std::io::BufReader::new(dataset);
    let mut white_king_positions = [0; 64];
    let mut black_king_positions = [0; 64];
    let mut pieces_on_board = [0; 33];
    let mut pieces_on_board_by_type = [0; 6];
    let mut movecount = [0; 2048];
    let mut win_draw_loss = [0; 3];
    while count != 0 {
        let mut value = PackedBoard::zeroed();
        reader.read_exact(bytemuck::bytes_of_mut(&mut value))?;
        let (board, _eval, wdl, _extra) = value.unpack().expect("invalid board");
        let wk_pos = board.king(Color::White);
        let bk_pos = board.king(Color::Black);
        white_king_positions[wk_pos as usize] += 1;
        black_king_positions[bk_pos as usize] += 1;
        pieces_on_board[board.occupied().len()] += 1;
        for sq in board.occupied() {
            let ty = board.piece_on(sq).unwrap();
            pieces_on_board_by_type[ty as usize] += 1;
        }
        movecount[board.fullmove_number() as usize] += 1;
        win_draw_loss[wdl as usize] += 1;
        count -= 1;
        if count & 0xFFFFF == 0 {
            let progress = positions - count;
            let proportion = progress as f64 / positions as f64;
            print!("\r\x1B[K{progress:12}/{positions} ({:4.1}%)", proportion * 100.0);
            let _ = std::io::stdout().flush();
        }
    }
    println!();
    println!("{} positions", positions);

    println!("Pieces on board:");
    let mean = pieces_on_board.iter().enumerate().map(|(i, v)| i * v).sum::<usize>() as f64 / positions as f64;
    println!("  Mean: {:.2}", mean);
    let mut for_sorting = pieces_on_board.iter().copied().enumerate().collect::<Vec<_>>();
    for_sorting.sort_by_key(|(_, v)| *v);
    let median = for_sorting[for_sorting.len() / 2];
    println!("  Median: {}", median.0);
    println!("  Distribution:");
    let mut left_column = Vec::with_capacity(15);
    let mut right_column = Vec::with_capacity(15);
    for (i, v) in pieces_on_board.iter().enumerate().skip(3) {
        let pcnt = *v as f64 / positions as f64 * 100.0;
        let s = format!("{:2}: {} ({:.1}%)", i, v, pcnt);
        if i < 17 {
            left_column.push(s);
        } else {
            right_column.push(s);
        }
    }
    for (l, r) in left_column.into_iter().zip(right_column.into_iter()) {
        println!("  {:<40}{}", l, r);
    }

    let mean_mvcnt = movecount.iter().enumerate().map(|(i, v)| i * v).sum::<usize>() as f64 / positions as f64;
    println!("Mean movecount: {:.2}", mean_mvcnt);

    println!("Win/draw/loss:");
    println!("  Win:  {:.2}%", win_draw_loss[0] as f64 / positions as f64 * 100.0);
    println!("  Draw: {:.2}%", win_draw_loss[1] as f64 / positions as f64 * 100.0);
    println!("  Loss: {:.2}%", win_draw_loss[2] as f64 / positions as f64 * 100.0);

    println!("White king positions:");
    let (idx, max) = white_king_positions.iter().enumerate().max_by_key(|(_, v)| **v).unwrap();
    let idx_sq = Square::index(idx);
    let pcnt = *max as f64 / positions as f64 * 100.0;
    println!("  Most: {} on {} ({:.3}%)", max, idx_sq, pcnt);
    let (idx, min) = white_king_positions.iter().enumerate().min_by_key(|(_, v)| **v).unwrap();
    let idx_sq = Square::index(idx);
    let pcnt = *min as f64 / positions as f64 * 100.0;
    println!("  Least: {} on {} ({:.3}%)", min, idx_sq, pcnt);
    println!("Black king positions:");
    let (idx, max) = black_king_positions.iter().enumerate().max_by_key(|(_, v)| **v).unwrap();
    let idx_sq = Square::index(idx);
    let pcnt = *max as f64 / positions as f64 * 100.0;
    println!("  Most: {} on {} ({:.3}%)", max, idx_sq, pcnt);
    let (idx, min) = black_king_positions.iter().enumerate().min_by_key(|(_, v)| **v).unwrap();
    let idx_sq = Square::index(idx);
    let pcnt = *min as f64 / positions as f64 * 100.0;
    println!("  Least: {} on {} ({:.3}%)", min, idx_sq, pcnt);

    Ok(())
}