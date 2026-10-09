use sonium_solvers_core::array;

#[test]
fn matrix_elements_are_evaluated_once_in_row_order() {
    let mut calls = 0;
    let mut next = || {
        calls += 1;
        calls as f64
    };
    let matrix = array![[next(), next()], [next(), next()], [next(), next()]];
    assert_eq!(calls, 6);
    assert_eq!(matrix.as_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn single_row_and_vector_elements_are_evaluated_once() {
    let mut calls = 0;
    let row = array![[
        {
            calls += 1;
            3.0
        },
        {
            calls += 1;
            4.0
        }
    ]];
    assert_eq!(calls, 2);
    assert_eq!(row.as_slice(), &[3.0, 4.0]);
    let vector = array![
        {
            calls += 1;
            5.0
        },
        {
            calls += 1;
            6.0
        }
    ];
    assert_eq!(calls, 4);
    assert_eq!(vector, vec![5.0, 6.0]);
}
