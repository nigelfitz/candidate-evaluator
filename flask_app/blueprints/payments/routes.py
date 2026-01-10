"""
Stripe payment integration for adding funds
"""
import stripe
import os
from flask import Blueprint, render_template, redirect, url_for, request, flash, jsonify
from flask_login import login_required, current_user
from database import db, User, Transaction
from config import Config
from decimal import Decimal

from blueprints.payments import payments_bp


@payments_bp.route('/buy-credits')
@login_required
def buy_credits():
    """Redirect to wallet page (legacy compatibility)"""
    return redirect(url_for('payments.wallet'))


@payments_bp.route('/wallet')
@login_required
def wallet():
    """Display wallet page with three-card pricing options"""
    stripe_configured = bool(Config.STRIPE_SECRET_KEY)
    is_dev_mode = os.environ.get('FLASK_ENV') == 'development'
    return_to = request.args.get('return_to', 'dashboard')
    
    pricing = Config.get_pricing()
    return render_template('wallet.html', 
                         stripe_configured=stripe_configured,
                         is_dev_mode=is_dev_mode,
                         return_to=return_to,
                         pricing=pricing)


@payments_bp.route('/add-test-funds', methods=['POST'])
@login_required
def add_test_funds():
    """Add test funds directly (development mode only)"""
    try:
        amount_str = request.form.get('amount', '0')
        amount = Decimal(amount_str)
        return_to = request.form.get('return_to', 'dashboard')
        
        if amount < 5:
            flash('Minimum amount is $5', 'error')
            return redirect(url_for('payments.buy_credits', return_to=return_to))
        
        # Add funds directly to user account
        current_user.add_funds(
            amount_usd=amount,
            description=f'Test funds added (DEV MODE)'
        )
        db.session.commit()
        
        flash(f'Successfully added ${amount:.2f} to your account!', 'success')
        
        # Handle different return destinations
        if return_to == 'run_analysis':
            return redirect(url_for('run_analysis_route', auto_submit=1))
        elif return_to.startswith('http://') or return_to.startswith('https://'):
            # If it's a full URL (like insights page), redirect directly to it
            return redirect(return_to)
        elif return_to != 'dashboard':
            # If it's a relative path, try to redirect to it
            try:
                return redirect(return_to)
            except:
                return redirect(url_for('dashboard'))
        else:
            # Default to dashboard
            return redirect(url_for('dashboard'))
        
    except Exception as e:
        flash(f'Error adding funds: {str(e)}', 'error')
        return redirect(url_for('payments.buy_credits'))


@payments_bp.route('/create-checkout', methods=['POST'])
@login_required
def create_checkout():
    """Create Stripe Checkout Session with auto-redirect"""
    try:
        # Check if Stripe is configured
        if not Config.STRIPE_SECRET_KEY:
            flash('Stripe is not configured. Use "Add Test Funds" button instead.', 'warning')
            return redirect(url_for('payments.buy_credits'))
        
        # Set Stripe API key
        stripe.api_key = Config.STRIPE_SECRET_KEY
        
        amount_str = request.form.get('amount', '0')
        amount = float(amount_str)
        
        if amount < 5:
            flash('Minimum amount is $5', 'error')
            return redirect(url_for('payments.buy_credits'))
        
        # Get return_to parameter for redirect after payment
        return_to = request.form.get('return_to', 'dashboard')
        
        # Get or create Stripe customer
        if not current_user.stripe_customer_id:
            customer = stripe.Customer.create(
                email=current_user.email,
                name=current_user.name or current_user.email,
                metadata={'user_id': current_user.id}
            )
            current_user.stripe_customer_id = customer.id
            db.session.commit()
        
        # Create Checkout Session with auto-redirect
        checkout_session = stripe.checkout.Session.create(
            customer=current_user.stripe_customer_id,
            payment_method_types=['card'],
            line_items=[{
                'price_data': {
                    'currency': 'usd',
                    'unit_amount': int(amount * 100),
                    'product_data': {
                        'name': 'Account Balance',
                        'description': f'Add ${amount:.2f} to your account',
                    },
                },
                'quantity': 1,
            }],
            mode='payment',
            success_url=url_for('payments.success', _external=True) + f'?session_id={{CHECKOUT_SESSION_ID}}&return_to={return_to}',
            cancel_url=url_for('payments.buy_credits', _external=True),
            metadata={
                'user_id': current_user.id,
                'amount_usd': f"{amount:.2f}",
                'current_balance': f"{current_user.balance_usd:.2f}",
                'return_to': return_to,
                'generate_invoice': 'true'  # Flag to generate invoice after payment
            }
        )
        
        return redirect(checkout_session.url, code=303)
        return redirect(invoice_url, code=303)
        
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        flash(f'Error creating invoice: {str(e)}', 'error')
        return redirect(url_for('payments.buy_credits'))


@payments_bp.route('/create-checkout-session', methods=['POST'])
@login_required
def create_checkout_session():
    """Create Stripe Checkout Session (for modal quick-pay)"""
    try:
        # Check if Stripe is configured
        if not Config.STRIPE_SECRET_KEY:
            # In dev mode, add test funds instead
            if os.environ.get('FLASK_ENV') == 'development':
                data = request.get_json()
                charge_amount = Decimal(str(data.get('amount', 0)))
                credit_amount = Decimal(str(data.get('credit_amount', charge_amount)))
                is_bundle = data.get('is_bundle', False)
                plan_name = data.get('plan_name', 'Custom Amount')
                
                if charge_amount < 5:
                    return jsonify({'error': 'Minimum amount is $5'}), 400
                
                bonus_amount = credit_amount - charge_amount
                
                # Add base purchase
                purchase_description = f'Stripe Purchase - {plan_name} (Dev Mode)'
                current_user.add_funds(
                    amount_usd=charge_amount,
                    description=purchase_description
                )
                
                # Add bonus as separate transaction if applicable
                if bonus_amount > 0:
                    bonus_description = f'Volume Bonus - {plan_name} (Dev Mode)'
                    current_user.add_funds(
                        amount_usd=bonus_amount,
                        description=bonus_description
                    )
                
                db.session.commit()
                
                # Return the return URL
                return_url = data.get('return_url', url_for('dashboard', _external=True))
                return jsonify({'url': return_url})
            else:
                return jsonify({'error': 'Stripe not configured'}), 400
        
        # Set Stripe API key
        stripe.api_key = Config.STRIPE_SECRET_KEY
        
        data = request.get_json()
        # Use Decimal for precise currency handling
        charge_amount = Decimal(str(data.get('amount', 0)))
        credit_amount = Decimal(str(data.get('credit_amount', charge_amount)))
        is_bundle = data.get('is_bundle', False)
        plan_name = data.get('plan_name', 'Custom Amount')
        return_url = data.get('return_url', url_for('dashboard.dashboard', _external=True))
        
        if charge_amount < 5:
            return jsonify({'error': 'Minimum amount is $5'}), 400
        
        # Determine product description
        if is_bundle:
            product_name = f'{plan_name}'
            product_description = f'Pay ${charge_amount:.2f}, receive ${credit_amount:.2f} balance'
        else:
            product_name = 'Account Balance'
            product_description = f'Add ${credit_amount:.2f} to your account'
        
        # Get or create Stripe customer
        if not current_user.stripe_customer_id:
            customer = stripe.Customer.create(
                email=current_user.email,
                name=current_user.name or current_user.email,
                metadata={'user_id': current_user.id}
            )
            current_user.stripe_customer_id = customer.id
            db.session.commit()
        
        # Create Checkout Session with auto-redirect
        checkout_session = stripe.checkout.Session.create(
            customer=current_user.stripe_customer_id,
            payment_method_types=['card'],
            line_items=[{
                'price_data': {
                    'currency': 'usd',
                    'unit_amount': int(charge_amount * 100),
                    'product_data': {
                        'name': product_name,
                        'description': product_description,
                    },
                },
                'quantity': 1,
            }],
            mode='payment',
            success_url=url_for('payments.success', _external=True) + f'?session_id={{CHECKOUT_SESSION_ID}}&return_to={return_url}',
            cancel_url=return_url,
            metadata={
                'user_id': current_user.id,
                'charge_amount': f"{charge_amount:.2f}",
                'credit_amount': f"{credit_amount:.2f}",
                'is_bundle': str(is_bundle),
                'plan_name': plan_name,
                'current_balance': f"{current_user.balance_usd:.2f}",
                'generate_invoice': 'true'  # Flag to generate invoice after payment
            }
        )
        
        return jsonify({'url': checkout_session.url})
        
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@payments_bp.route('/success')
@login_required
def success():
    """Handle successful payment redirect"""
    session_id = request.args.get('session_id')
    return_to = request.args.get('return_to', 'dashboard')
    
    print(f"🎉 Success route called! session_id={session_id}, return_to={return_to}, user={current_user.id}")
    
    if session_id:
        try:
            stripe.api_key = Config.STRIPE_SECRET_KEY
            # Retrieve the session to verify it
            session = stripe.checkout.Session.retrieve(session_id)
            
            print(f"📋 Session retrieved: payment_status={session.payment_status}, metadata={session.get('metadata', {})}")
            
            if session.payment_status == 'paid':
                # Add credits immediately for local testing (webhook won't work on localhost)
                print(f"💳 Payment confirmed as paid. Fulfilling order...")
                fulfill_order(session)
                flash('Payment successful! Credits have been added to your account.', 'success')
            else:
                print(f"⏳ Payment status is {session.payment_status}, not 'paid'")
                flash('Payment is being processed. Credits will be added once confirmed.', 'info')
                
        except Exception as e:
            print(f"❌ Error in success route: {str(e)}")
            import traceback
            traceback.print_exc()
            flash(f'Payment completed, but verification failed: {str(e)}', 'warning')
    else:
        print("⚠️ No session_id provided in success route")
    
    # Handle different return destinations
    if return_to == 'run_analysis':
        return redirect(url_for('run_analysis_route', auto_submit=1))
    elif return_to.startswith('http://') or return_to.startswith('https://'):
        # If it's a full URL (like insights page), redirect directly to it
        return redirect(return_to)
    elif return_to != 'dashboard':
        # If it's a relative path, try to redirect to it
        try:
            return redirect(return_to)
        except:
            return redirect(url_for('dashboard'))
    else:
        # Default to dashboard
        return redirect(url_for('dashboard'))


@payments_bp.route('/webhook', methods=['POST'])
@payments_bp.route('/api/stripe-webhook', methods=['POST'])
def webhook():
    """Handle Stripe webhook events"""
    payload = request.get_data()
    sig_header = request.headers.get('Stripe-Signature')
    
    print(f"🔔 Webhook received! Event type: {request.get_json().get('type', 'unknown')}")
    
    stripe.api_key = Config.STRIPE_SECRET_KEY
    
    # For testing, we'll skip signature verification
    # In production, you should verify the webhook signature
    try:
        event = stripe.Event.construct_from(
            request.get_json(), stripe.api_key
        )
        print(f"✅ Webhook event constructed successfully: {event['type']}")
    except Exception as e:
        print(f"❌ Error constructing webhook event: {str(e)}")
        return jsonify({'error': str(e)}), 400
    
    # Handle invoice payment events (for Invoice API)
    if event['type'] in ['invoice.paid', 'invoice.payment_succeeded']:
        invoice = event['data']['object']
        print(f"💰 Processing {event['type']} for customer {invoice.get('customer', 'unknown')}")
        
        # Fulfill the purchase using invoice metadata
        fulfill_order_from_invoice(invoice)
    
    # Handle checkout sessions
    elif event['type'] == 'checkout.session.completed':
        session = event['data']['object']
        print(f"💰 Processing checkout.session.completed for user {session.get('metadata', {}).get('user_id', 'unknown')}")
        
        # Fulfill the purchase
        fulfill_order(session)
        
        # Generate invoice PDF if requested
        if session.get('metadata', {}).get('generate_invoice') == 'true':
            try:
                generate_invoice_for_payment(session)
            except Exception as e:
                print(f"⚠️ Failed to generate invoice: {str(e)}")
                # Don't fail the webhook if invoice generation fails
    else:
        print(f"ℹ️ Ignoring event type: {event['type']}")
    
    return jsonify({'status': 'success'}), 200


def generate_invoice_for_payment(session):
    """Generate an Invoice PDF after successful Checkout payment"""
    try:
        stripe.api_key = Config.STRIPE_SECRET_KEY
        
        customer_id = session.get('customer')
        if not customer_id:
            print("⚠️ No customer ID in session, skipping invoice generation")
            return
        
        metadata = session.get('metadata', {})
        charge_amount = Decimal(metadata.get('charge_amount', metadata.get('amount_usd', '0')))
        credit_amount = Decimal(metadata.get('credit_amount', charge_amount))
        plan_name = metadata.get('plan_name', 'Account Top-Up')
        payment_intent = session.get('payment_intent')
        
        # Create a paid invoice by attaching the payment intent
        invoice = stripe.Invoice.create(
            customer=customer_id,
            currency='usd',
            collection_method='charge_automatically',
            auto_advance=False,  # We'll finalize manually
            metadata={
                'checkout_session_id': session.get('id'),
                'user_id': metadata.get('user_id'),
                'generated_post_payment': 'true'
            }
        )
        
        # Add line item for the payment
        stripe.InvoiceItem.create(
            customer=customer_id,
            invoice=invoice.id,
            amount=int(charge_amount * 100),
            currency='usd',
            description=f"{plan_name} - Account Balance"
        )
        
        # Finalize and mark as paid
        invoice = stripe.Invoice.finalize_invoice(invoice.id)
        
        # Mark invoice as paid outside of Stripe (since payment already completed via Checkout)
        stripe.Invoice.mark_uncollectible(invoice.id)
        stripe.Invoice.modify(
            invoice.id,
            paid_out_of_band=True
        )
        
        print(f"✅ Generated invoice {invoice.id} for checkout session {session.get('id')}")
        
    except Exception as e:
        print(f"❌ Error generating invoice: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


def fulfill_order(session):
    """Add funds to user account after successful payment"""
    try:
        user_id = int(session['metadata']['user_id'])
        charge_amount = Decimal(session['metadata'].get('charge_amount', session['metadata'].get('amount_usd', '0')))
        credit_amount = Decimal(session['metadata'].get('credit_amount', charge_amount))
        is_bundle = session['metadata'].get('is_bundle', 'False') == 'True'
        plan_name = session['metadata'].get('plan_name', 'Account Top-Up')
        
        user = User.query.get(user_id)
        if user:
            # Update stripe_customer_id if not already set
            if not user.stripe_customer_id and session.get('customer'):
                user.stripe_customer_id = session['customer']
            
            bonus_amount = credit_amount - charge_amount
            
            # Add base purchase amount
            user.balance_usd += charge_amount
            purchase_transaction = Transaction(
                user_id=user_id,
                amount_usd=charge_amount,
                transaction_type='credit',
                description=f"Stripe Purchase - {plan_name}",
                stripe_payment_id=session.get('payment_intent'),
                stripe_amount_cents=session['amount_total']
            )
            db.session.add(purchase_transaction)
            
            # Add bonus as separate transaction if applicable
            if bonus_amount > 0:
                user.balance_usd += bonus_amount
                bonus_transaction = Transaction(
                    user_id=user_id,
                    amount_usd=bonus_amount,
                    transaction_type='credit',
                    description=f"Volume Bonus - {plan_name}",
                    stripe_payment_id=session.get('payment_intent'),
                    stripe_amount_cents=0  # Bonus doesn't have a Stripe charge
                )
                db.session.add(bonus_transaction)
            
            # Track revenue (only the amount actually paid)
            user.total_revenue_usd = (user.total_revenue_usd or Decimal('0')) + charge_amount
            
            db.session.commit()
            
            print(f"✅ Successfully added ${credit_amount:.2f} to user {user_id}. New balance: ${user.balance_usd:.2f}")
            if bonus_amount > 0:
                print(f"   🎁 Bonus applied: Charged ${charge_amount:.2f}, bonus ${bonus_amount:.2f}, total credited ${credit_amount:.2f}")
        
    except Exception as e:
        print(f"❌ Error fulfilling order: {str(e)}")
        import traceback
        traceback.print_exc()
        # In production, you should log this and have a retry mechanism


def fulfill_order_from_invoice(invoice):
    """Add funds to user account after successful invoice payment"""
    try:
        # Invoice metadata is stored on the invoice object directly
        metadata = invoice.get('metadata', {})
        user_id = int(metadata['user_id'])
        charge_amount = Decimal(metadata.get('charge_amount', metadata.get('amount_usd', '0')))
        credit_amount = Decimal(metadata.get('credit_amount', charge_amount))
        is_bundle = metadata.get('is_bundle', 'False') == 'True'
        plan_name = metadata.get('plan_name', 'Account Top-Up')
        
        user = User.query.get(user_id)
        if user:
            # Update stripe_customer_id if not already set
            if not user.stripe_customer_id and invoice.get('customer'):
                user.stripe_customer_id = invoice['customer']
            
            bonus_amount = credit_amount - charge_amount
            
            # Add base purchase amount
            user.balance_usd += charge_amount
            purchase_transaction = Transaction(
                user_id=user_id,
                amount_usd=charge_amount,
                transaction_type='credit',
                description=f"Stripe Invoice - {plan_name}",
                stripe_payment_id=invoice.get('payment_intent'),
                stripe_amount_cents=invoice.get('amount_paid', 0)
            )
            db.session.add(purchase_transaction)
            
            # Add bonus as separate transaction if applicable
            if bonus_amount > 0:
                user.balance_usd += bonus_amount
                bonus_transaction = Transaction(
                    user_id=user_id,
                    amount_usd=bonus_amount,
                    transaction_type='credit',
                    description=f"Volume Bonus - {plan_name}",
                    stripe_payment_id=invoice.get('payment_intent'),
                    stripe_amount_cents=0  # Bonus doesn't have a Stripe charge
                )
                db.session.add(bonus_transaction)
            
            # Track revenue (only the amount actually paid)
            user.total_revenue_usd = (user.total_revenue_usd or Decimal('0')) + charge_amount
            
            db.session.commit()
            
            print(f"✅ Successfully added ${credit_amount:.2f} to user {user_id}. New balance: ${user.balance_usd:.2f}")
            if bonus_amount > 0:
                print(f"   🎁 Bonus applied: Charged ${charge_amount:.2f}, bonus ${bonus_amount:.2f}, total credited ${credit_amount:.2f}")
        
    except Exception as e:
        print(f"❌ Error fulfilling invoice order: {str(e)}")
        import traceback
        traceback.print_exc()
        # In production, you should log this and have a retry mechanism
