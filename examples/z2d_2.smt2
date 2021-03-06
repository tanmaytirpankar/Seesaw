;"""
;_F3 = [-0.06868482029115039, 0.06361361011821207];
;_F2 = [2.2255636349853907, 2.825204175582711];
;_F1 = [-0.002206520729082753, 0.003475337353143628];
;px0 = [21.76, 21.78];
;py0 = [21.76, 21.78];
;pz0 = [19.999, 20.001];
;px1 = [22.017, 22.027];
;py1 = [20.96, 21.96];
;pz1 = [19.99, 20.01];
;px2 = [20.94, 21.24];
;py2 = [20.84, 21.14];
;pz2 = [21.99, 22.05];
;((px0 - 20.0)^2 + (py0 - 20.0)^2 + (pz0 - 20.0)^2 - 3.46944695195361e-16 < 6.25) && (((px1 - 20.0)^2 + (py1 - 20.0)^2 + (pz1 - 20.0)^2 + 7.87700190430938e-16 > 6.24999999999999) || ((-_F1*(px0 - 20.0) + px1 - 20.0)^2 + (-_F1*(py0 - 20.0) + py1 - 20.0)^2 + (-_F1*(pz0 - 20.0) + pz1 - 20.0)^2 + 7.87700190430938e-16 > 6.24999999999999) || ((px1 - 20.0)^2 + (py1 - 20.0)^2 + (pz1 - 20.0)^2 + 7.87700190430938e-16 > 0.25*(sqrt((px0 - 20.0)^2 + (py0 - 20.0)^2 + (pz0 - 20.0)^2) + 2.5)^2 - 6.68275248813409e-15) || ((-_F1*(px0 - 20.0) + px1 - 20.0)^2 + (-_F1*(py0 - 20.0) + py1 - 20.0)^2 + (-_F1*(pz0 - 20.0) + pz1 - 20.0)^2 + 7.87700190430938e-16 > 0.25*(sqrt((px0 - 20.0)^2 + (py0 - 20.0)^2 + (pz0 - 20.0)^2) + 2.5)^2 - 6.68275248813409e-15));
;"""

(set-logic QF_NRA)

(declare-fun px0 () Real)
(declare-fun py0 () Real)
(declare-fun pz0 () Real)
(declare-fun px1 () Real)
(declare-fun py1 () Real)
(declare-fun pz1 () Real)
(declare-fun px2 () Real)
(declare-fun py2 () Real)
(declare-fun pz2 () Real)
(declare-fun _F1 () Real)
(declare-fun _F2 () Real)
(declare-fun _F3 () Real)
(assert (and (>= px0 (/ 544.0 25.0)) (<= px0 (/ 1089.0 50.0))))
(assert (and (>= py0 (/ 544.0 25.0)) (<= py0 (/ 1089.0 50.0))))
(assert (and (>= pz0 (/ 19999.0 1000.0)) (<= pz0 (/ 20001.0 1000.0))))
(assert (and (>= px1 (/ 22017.0 1000.0)) (<= px1 (/ 22027.0 1000.0))))
(assert (and (>= py1 (/ 524.0 25.0)) (<= py1 (/ 549.0 25.0))))
(assert (and (>= pz1 (/ 1999.0 100.0)) (<= pz1 (/ 2001.0 100.0))))
(assert (and (>= px2 (/ 1047.0 50.0)) (<= px2 (/ 531.0 25.0))))
(assert (and (>= py2 (/ 521.0 25.0)) (<= py2 (/ 1057.0 50.0))))
(assert (and (>= pz2 (/ 2199.0 100.0)) (<= pz2 (/ 441.0 20.0))))
(assert (and (>= _F1 (- (/ 2206520729082753.0 1000000000000000000.0)))
     (<= _F1 (/ 868834338285907.0 250000000000000000.0))))
(assert (and (>= _F2 (/ 22255636349853907.0 10000000000000000.0))
     (<= _F2 (/ 2825204175582711.0 1000000000000000.0))))
(assert (and (>= _F3 (- (/ 6868482029115039.0 100000000000000000.0)))
     (<= _F3 (/ 6361361011821207.0 100000000000000000.0))))
(assert (let ((a!1 (- (+ (^ (- px0 20.0) 2.0) (^ (- py0 20.0) 2.0) (^ (- pz0 20.0) 2.0))
              (/ 346944695195361.0 1000000000000000000000000000000.0))))
  (> a!1 (/ 25.0 4.0))))
(assert (let ((a!1 (+ (^ (- px1 20.0) 2.0)
              (^ (- py1 20.0) 2.0)
              (^ (- pz1 20.0) 2.0)
              (/ 393850095215469.0 500000000000000000000000000000.0)))
      (a!2 (- (+ (* (- _F1) (- px0 20.0)) px1) 20.0))
      (a!3 (- (+ (* (- _F1) (- py0 20.0)) py1) 20.0))
      (a!4 (- (+ (* (- _F1) (- pz0 20.0)) pz1) 20.0))
      (a!6 (^ (+ (^ (- px0 20.0) 2.0) (^ (- py0 20.0) 2.0) (^ (- pz0 20.0) 2.0))
              (/ 1.0 2.0))))
(let ((a!5 (+ (^ a!2 2.0)
              (^ a!3 2.0)
              (^ a!4 2.0)
              (/ 393850095215469.0 500000000000000000000000000000.0)))
      (a!7 (- (* (/ 1.0 4.0) (^ (+ a!6 (/ 5.0 2.0)) 2.0))
              (/ 668275248813409.0 100000000000000000000000000000.0))))
  (or (> a!1 (/ 624999999999999.0 100000000000000.0))
      (> a!5 (/ 624999999999999.0 100000000000000.0))
      (> a!1 a!7)
      (> a!5 a!7)))))
(check-sat)
(get-model)
(exit)
