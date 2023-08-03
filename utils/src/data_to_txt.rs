use std::io::{BufReader, BufWriter, Result, Write};
use std::path::PathBuf;
use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
};

use bytemuck::Zeroable;
use cozy_chess::Board;
use marlinformat::PackedBoard;
use structopt::StructOpt;

/// Convert marlinformat to text data format.
#[derive(StructOpt)]
pub struct Options {
    #[structopt(short, long)]
    output: PathBuf,

    data_file: PathBuf,

    #[structopt(short, long)]
    format: String,

    #[structopt(short, long)]
    limit: Option<u64>,
}

trait Format {
    fn write_into(
        board: &Board,
        cp: i16,
        wdl: u8,
        extra: u8,
        output: &mut impl Write,
    ) -> Result<()>;
}

struct Legacy;
impl Format for Legacy {
    fn write_into(
        board: &Board,
        cp: i16,
        wdl: u8,
        _extra: u8,
        output: &mut impl Write,
    ) -> Result<()> {
        let wdl = match wdl {
            0 => "0.0",
            1 => "0.5",
            2 => "1.0",
            _ => unreachable!(),
        };
        writeln!(output, "{board} | {cp} | {wdl}")
    }
}
struct Cudad;
impl Format for Cudad {
    fn write_into(
        board: &Board,
        cp: i16,
        wdl: u8,
        _extra: u8,
        output: &mut impl Write,
    ) -> Result<()> {
        let wdl = match wdl {
            0 => "0.0",
            1 => "0.5",
            2 => "1.0",
            _ => unreachable!(),
        };
        writeln!(output, "{board} [{wdl}] {cp}")
    }
}

pub fn run(options: Options) -> Result<()> {
    let mut data = std::fs::File::open(options.data_file)?;
    let size_bytes = data.seek(SeekFrom::End(0))?;
    data.seek(SeekFrom::Start(0))?;
    let count = size_bytes / std::mem::size_of::<PackedBoard>() as u64;
    let count = options.limit.map_or(count, |limit| limit.min(count));

    let mut reader = BufReader::new(data);

    let mut output = BufWriter::new(File::create(options.output)?);

    match options.format.as_str() {
        "legacy" => conversion_loop::<Legacy>(count, &mut reader, &mut output)?,
        "cudad" => conversion_loop::<Cudad>(count, &mut reader, &mut output)?,
        _ => panic!(
            "unknown format {}, valid formats are legacy and cudad",
            options.format
        ),
    }

    Ok(())
}

fn conversion_loop<F: Format>(
    count: u64,
    reader: &mut impl Read,
    output: &mut impl Write,
) -> Result<()> {
    print!("at 0/{count}\r");
    for pos in 0..count {
        let mut value = PackedBoard::zeroed();
        reader.read_exact(bytemuck::bytes_of_mut(&mut value))?;
        let (board, cp, wdl, extra) = value.unpack().expect("invalid board");
        F::write_into(&board, cp, wdl, extra, output)?;
        if pos % 1000 == 0 {
            print!("at {pos}/{count}\r");
        }
    }
    println!("at {count}/{count}");

    Ok(())
}
