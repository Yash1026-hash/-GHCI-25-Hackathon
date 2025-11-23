document.addEventListener('DOMContentLoaded', () => {
    const connectBtn = document.getElementById('connectWalletBtn');
    const loadingModal = document.getElementById('loadingModal');
    const historyTableBody = document.querySelector('#historyTable tbody');
    const txnBadge = document.getElementById('txnBadge');
    const totalSpendEl = document.getElementById('totalSpend');

    // Sidebar Navigation
    const navItems = document.querySelectorAll('.nav-item');
    const viewSections = document.querySelectorAll('.view-section');

    navItems.forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();

            // Remove active class from all nav items
            navItems.forEach(nav => nav.classList.remove('active'));

            // Add active class to clicked item
            item.classList.add('active');

            // Hide all views
            viewSections.forEach(view => view.classList.add('hidden'));
            viewSections.forEach(view => view.classList.remove('active'));

            // Show target view
            const targetId = item.getAttribute('data-target');
            const targetView = document.getElementById(targetId);
            if (targetView) {
                targetView.classList.remove('hidden');
                targetView.classList.add('active');
            }
        });
    });

    // Connect Wallet Handler
    if (connectBtn) {
        connectBtn.addEventListener('click', async () => {
            console.log('Connect Wallet button clicked');

            // Show loading
            loadingModal.classList.remove('hidden');
            console.log('Modal shown');

            // Safety timeout
            const safetyTimeout = setTimeout(() => {
                if (!loadingModal.classList.contains('hidden')) {
                    console.warn('Connection timed out - forcing close');
                    loadingModal.classList.add('hidden');
                    alert('Connection timed out. Please try again.');
                }
            }, 8000);

            try {
                // 1. Simulate Connection
                console.log('Calling /api/connect_wallet...');
                const connectRes = await fetch('/api/connect_wallet', { method: 'POST' });
                console.log('Connect response:', connectRes.status);

                if (!connectRes.ok) throw new Error('Connection failed');

                // 2. Fetch History
                console.log('Calling /api/history...');
                const response = await fetch('/api/history');
                console.log('History response:', response.status);

                if (!response.ok) throw new Error('Failed to fetch history');

                const history = await response.json();
                console.log('History data:', history);

                // 3. Render History
                renderHistory(history);

                // 4. Update Stats
                const total = history.reduce((sum, item) => sum + item.amount, 0);
                totalSpendEl.textContent = total.toFixed(2);

                // Change button state
                connectBtn.innerHTML = '<i class="fa-solid fa-check"></i> Connected';
                connectBtn.style.background = 'var(--success)';
                connectBtn.disabled = true;

            } catch (error) {
                console.error('Error during connection:', error);
                alert('Failed to connect wallet: ' + error.message);
            } finally {
                // Always hide loading
                console.log('Cleaning up...');
                clearTimeout(safetyTimeout);
                loadingModal.classList.add('hidden');
            }
        });
    }

    // Manual Categorization Handler
    const form = document.getElementById('transactionForm');
    if (form) {
        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            const rawString = document.getElementById('rawString').value;
            const amount = document.getElementById('amount').value;

            try {
                const response = await fetch('/api/categorize', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ raw_string: rawString, amount: amount })
                });
                const data = await response.json();

                // Show Result
                const resultDiv = document.getElementById('singleResult');
                resultDiv.classList.remove('hidden');

                document.getElementById('resultCategory').textContent = data.category;
                document.getElementById('confidenceFill').style.width = (data.confidence * 100) + '%';
                document.getElementById('explanationText').textContent = `Identified as ${data.category} based on merchant "${data.merchant_name}"`;
            } catch (error) {
                console.error('Categorization error:', error);
                alert('Failed to categorize transaction');
            }
        });
    }
});

function renderHistory(transactions) {
    const tbody = document.querySelector('#historyTable tbody');
    if (!tbody) return;

    tbody.innerHTML = ''; // Clear empty state

    transactions.forEach(txn => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>
                <div style="font-weight: 600;">${txn.merchant}</div>
            </td>
            <td>${txn.date}</td>
            <td>
                <span style="
                    background: rgba(59, 130, 246, 0.1); 
                    color: #60a5fa; 
                    padding: 0.25rem 0.75rem; 
                    border-radius: 1rem; 
                    font-size: 0.85rem;">
                    ${txn.category}
                </span>
            </td>
            <td>$${txn.amount.toFixed(2)}</td>
        `;
        tbody.appendChild(row);
    });

    const badge = document.getElementById('txnBadge');
    if (badge) badge.textContent = `${transactions.length} items`;
}
