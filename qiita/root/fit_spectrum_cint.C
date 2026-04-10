void fit_spectrum_cint(Int_t show_fit_result = 0)
{
    gStyle->SetOptStat(0);
    gStyle->SetOptFit(0);

    const Int_t NBIN = 200;
    const Double_t XMIN = 5870.0;
    const Double_t XMAX = 5920.0;

    TH1D *h = new TH1D("h", "Mock spectrum;Energy (eV);Counts", NBIN, XMIN, XMAX);

    // -----------------------------
    // 真のパラメータ
    // -----------------------------
    Double_t true_amp1   = 80.0;
    Double_t true_mu1    = 5890.0;
    Double_t true_sigma1 = 1.8;

    Double_t true_amp2   = 140.0;
    Double_t true_mu2    = 5900.0;
    Double_t true_sigma2 = 2.2;

    Double_t true_bkg0   = 30.0;
    Double_t true_bkg1   = -0.15;

    // -----------------------------
    // 擬似データ生成
    // -----------------------------
    TRandom3 rand(42);

    Int_t i;
    for (i = 1; i <= NBIN; i++) {
        Double_t x = h->GetBinCenter(i);
        Double_t xshift = x - 5870.0;

        Double_t g1 = true_amp1 * TMath::Exp(-0.5 * TMath::Power((x - true_mu1) / true_sigma1, 2.0));
        Double_t g2 = true_amp2 * TMath::Exp(-0.5 * TMath::Power((x - true_mu2) / true_sigma2, 2.0));
        Double_t bg = true_bkg0 + true_bkg1 * xshift;

        Double_t ytrue = g1 + g2 + bg;
        Double_t yobs  = rand.Poisson(ytrue);

        h->SetBinContent(i, yobs);

        if (yobs > 0.0) {
            h->SetBinError(i, TMath::Sqrt(yobs));
        } else {
            h->SetBinError(i, 1.0);
        }
    }

    // -----------------------------
    // フィット関数
    // -----------------------------
    TF1 *f = new TF1(
        "f",
        "gaus(0) + gaus(3) + [6] + [7]*(x-5870.0)",
        XMIN, XMAX
    );

    f->SetParNames("amp1", "mu1", "sigma1",
                   "amp2", "mu2", "sigma2",
                   "bkg0", "bkg1");

    f->SetParameters(60.0, 5889.0, 1.5,
                     120.0, 5901.0, 1.5,
                     20.0, -0.05);

    f->SetParLimits(2, 0.1, 10.0);
    f->SetParLimits(5, 0.1, 10.0);

    // -----------------------------
    // フィット実行
    // S : 詳細結果
    // E : MINOS 誤差
    // R : 関数範囲でフィット
    // L : ポアソン尤度
    // -----------------------------
    TFitResultPtr r = h->Fit("f", "S E R L");

    cout << "----------------------------------" << endl;
    cout << "Fit status = " << Int_t(r) << endl;
    cout << "Chi2       = " << r->Chi2() << endl;
    cout << "NDF        = " << r->Ndf() << endl;
    if (r->Ndf() != 0) {
        cout << "Chi2/NDF   = " << r->Chi2() / r->Ndf() << endl;
    }
    cout << "----------------------------------" << endl;

    Int_t npar = f->GetNpar();
    for (i = 0; i < npar; i++) {
        cout << f->GetParName(i)
             << " = " << r->Value(i)
             << "  +" << r->UpperError(i)
             << "  "  << r->LowerError(i)
             << endl;
    }

    // -----------------------------
    // pull ヒストグラム
    // pull = (data - model) / error
    // -----------------------------
    TH1D *h_pull = (TH1D*)h->Clone("h_pull");
    h_pull->Reset();
    h_pull->SetTitle(";Energy (eV);Pull");

    for (i = 1; i <= NBIN; i++) {
        Double_t x     = h->GetBinCenter(i);
        Double_t y     = h->GetBinContent(i);
        Double_t yerr  = h->GetBinError(i);
        Double_t yfit  = f->Eval(x);

        Double_t pull = 0.0;
        if (yerr > 0.0) {
            pull = (y - yfit) / yerr;
        }

        h_pull->SetBinContent(i, pull);
        h_pull->SetBinError(i, 0.0);
    }

    // -----------------------------
    // 描画
    // -----------------------------
    TCanvas *c1 = new TCanvas("c1", "fit with residual", 900, 800);

    TPad *pad1 = new TPad("pad1", "pad1", 0.0, 0.30, 1.0, 1.0);
    TPad *pad2 = new TPad("pad2", "pad2", 0.0, 0.00, 1.0, 0.30);

    pad1->SetBottomMargin(0.02);
    pad2->SetTopMargin(0.03);
    pad2->SetBottomMargin(0.30);

    pad1->Draw();
    pad2->Draw();

    // -----------------------------
    // 上段
    // -----------------------------
    pad1->cd();
    h->SetMarkerStyle(20);
    h->SetMarkerSize(0.8);
    h->GetXaxis()->SetLabelSize(0.0);
    h->GetXaxis()->SetTitleSize(0.0);
    h->Draw("E");

    f->SetLineColor(kRed);
    f->SetLineWidth(2);
    f->Draw("same");

    // フィット結果を書き込む（右上、小さめ、MINOS非対称誤差付き）
    if (show_fit_result != 0) {
        TLatex latex;
        latex.SetNDC();
        latex.SetTextSize(0.024);
        latex.SetTextAlign(33);

        // 色を青にする
        latex.SetTextColor(kBlue);

        Double_t x0 = 0.97;
        Double_t y0 = 0.93;
        Double_t dy = 0.037;

        Double_t chi2ndf = 0.0;
        if (r->Ndf() != 0) {
            chi2ndf = r->Chi2() / r->Ndf();
        }

        latex.DrawLatex(x0, y0 - 0*dy,
                        Form("#chi^{2}/NDF = %.1f / %d = %.2f",
                             r->Chi2(), r->Ndf(), chi2ndf));

        latex.DrawLatex(x0, y0 - 1*dy,
                        Form("amp1 = %.1f^{+%.1f}_{-%.1f}",
                             r->Value(0), r->UpperError(0), -r->LowerError(0)));

        latex.DrawLatex(x0, y0 - 2*dy,
                        Form("#mu_{1} = %.2f^{+%.2f}_{-%.2f}",
                             r->Value(1), r->UpperError(1), -r->LowerError(1)));

        latex.DrawLatex(x0, y0 - 3*dy,
                        Form("#sigma_{1} = %.2f^{+%.2f}_{-%.2f}",
                             r->Value(2), r->UpperError(2), -r->LowerError(2)));

        latex.DrawLatex(x0, y0 - 4*dy,
                        Form("amp2 = %.1f^{+%.1f}_{-%.1f}",
                             r->Value(3), r->UpperError(3), -r->LowerError(3)));

        latex.DrawLatex(x0, y0 - 5*dy,
                        Form("#mu_{2} = %.2f^{+%.2f}_{-%.2f}",
                             r->Value(4), r->UpperError(4), -r->LowerError(4)));

        latex.DrawLatex(x0, y0 - 6*dy,
                        Form("#sigma_{2} = %.2f^{+%.2f}_{-%.2f}",
                             r->Value(5), r->UpperError(5), -r->LowerError(5)));

        latex.DrawLatex(x0, y0 - 7*dy,
                        Form("bkg0 = %.2f^{+%.2f}_{-%.2f}",
                             r->Value(6), r->UpperError(6), -r->LowerError(6)));

        latex.DrawLatex(x0, y0 - 8*dy,
                        Form("bkg1 = %.4f^{+%.4f}_{-%.4f}",
                             r->Value(7), r->UpperError(7), -r->LowerError(7)));
    }

    // -----------------------------
    // 下段
    // -----------------------------
    pad2->cd();
    h_pull->SetMarkerStyle(20);
    h_pull->SetMarkerSize(0.6);
    h_pull->SetMinimum(-5.0);
    h_pull->SetMaximum(5.0);
    h_pull->GetXaxis()->SetTitleSize(0.10);
    h_pull->GetXaxis()->SetLabelSize(0.09);
    h_pull->GetYaxis()->SetTitleSize(0.08);
    h_pull->GetYaxis()->SetLabelSize(0.08);
    h_pull->GetYaxis()->SetTitleOffset(0.45);
    h_pull->Draw("P");

    TLine *line0 = new TLine(XMIN, 0.0, XMAX, 0.0);
    line0->SetLineColor(kBlue);
    line0->SetLineStyle(2);
    line0->Draw("same");

    c1->SaveAs("fit_spectrum_cint.png");
}