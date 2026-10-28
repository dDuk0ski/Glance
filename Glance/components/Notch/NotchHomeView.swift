//
//  NotchHomeView.swift
//  Glance
//
//  Originally from boring.notch (Hugo Persson et al.); music, calendar and
//  camera content removed. This is where the Glance prompt and response UI lives.
//

import SwiftUI

struct NotchHomeView: View {
    @EnvironmentObject var vm: BoringViewModel

    var body: some View {
        VStack(spacing: 6) {
            Image(systemName: "bubble.left.fill")
                .font(.title2)
                .foregroundStyle(.gray)
            Text("Glance")
                .font(.headline)
                .foregroundStyle(.white)
            Text("Prompt UI coming soon")
                .font(.caption)
                .foregroundStyle(.gray)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .transition(.asymmetric(insertion: .opacity.combined(with: .move(edge: .top)), removal: .opacity))
        .blur(radius: vm.notchState == .closed ? 30 : 0)
    }
}
