use structopt::StructOpt;

mod convert;
mod interleave;
mod shuffle;
mod txt_to_data;
mod data_to_txt;
mod count;
mod stats;

#[derive(StructOpt)]
pub enum Options {
    Convert(convert::Options),
    Shuffle(shuffle::Options),
    Interleave(interleave::Options),
    TxtToData(txt_to_data::Options),
    DataToTxt(data_to_txt::Options),
    Count(count::Options),
    Stats(stats::Options),
}

fn main() {
    match Options::from_args() {
        Options::Convert(options) => convert::run(options),
        Options::Shuffle(options) => shuffle::run(options).unwrap(),
        Options::Interleave(options) => interleave::run(options).unwrap(),
        Options::TxtToData(options) => txt_to_data::run(options).unwrap(),
        Options::DataToTxt(options) => data_to_txt::run(options).unwrap(),
        Options::Count(options) => count::run(options).unwrap(),
        Options::Stats(options) => stats::run(options).unwrap(),
    }
}
