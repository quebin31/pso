pub struct Fitness<'a, T> {
    func: Box<dyn Fn(&T) -> f64 + 'a>,
    minimization: bool,
}

impl<'a, T> Fitness<'a, T> {
    pub fn new<F>(func: F, minimization: bool) -> Self
    where
        F: Fn(&T) -> f64 + 'a,
    {
        Self {
            func: Box::new(func),
            minimization,
        }
    }

    pub fn calculate(&self, val: &T) -> f64 {
        (self.func)(val)
    }

    pub fn calculate_for_maximization(&self, val: &T) -> f64 {
        if self.minimization {
            -(self.func)(val)
        } else {
            (self.func)(val)
        }
    }

    pub fn is_minimization(&self) -> bool {
        self.minimization
    }
}
