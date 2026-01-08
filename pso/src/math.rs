pub fn mean(v: &[f32]) -> f32 {
    let sum: f32 = v.iter().sum();
    sum / v.len() as f32
}

pub fn median(v: &mut [f32]) -> f32 {
    v.sort_by(|a, b| a.total_cmp(b));
    let len = v.len();
    if len.is_multiple_of(2) {
        (v[len / 2 - 1] + v[len / 2]) / 2.0
    } else {
        v[len / 2]
    }
}

pub fn std_dev(v: &[f32]) -> f32 {
    let m = mean(v);
    let variance: f32 = v.iter().map(|x| (x - m).powi(2)).sum::<f32>() / v.len() as f32;
    variance.sqrt()
}
